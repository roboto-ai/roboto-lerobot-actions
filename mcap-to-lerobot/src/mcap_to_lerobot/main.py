from lerobot.datasets.lerobot_dataset import LeRobotDataset
import roboto

from .logger import logger
from .mcap import (
    McapTopic,
    calculate_fps_from_timestamps,
    find_lowest_frequency_topic,
    load_and_parse_mcap,
)
from .lerobot import generate_frames


def main(context: roboto.InvocationContext) -> None:
    """
    Convert MCAP files to LeRobot dataset format.

    Each MCAP file represents one episode with dual-arm robot data.
    Processes multiple episodes in a streaming fashion.
    """
    logger.setLevel(context.log_level)
    logger.info("Starting MCAP to LeRobot conversion")

    action_input = context.get_input()
    mcap_files = list(action_input.files)

    if not mcap_files:
        logger.error("No input files found")
        return

    # Process first file outside of primary processing loop to determine LeRobot meta
    first_file, _ = mcap_files[0]
    logger.info(
        "Processing episode 1/%d: %s", len(mcap_files), first_file.relative_path
    )

    if first_file.ingestion_status != roboto.IngestionStatus.Ingested:
        logger.error("First file is not ingested: %s", first_file.relative_path)
        return

    episode = load_and_parse_mcap(first_file)

    alignment_topic = context.get_optional_parameter("alignment_topic")
    if alignment_topic is not None:
        valid_topics = [
            McapTopic.ObservationStates.value,
            McapTopic.Action.value,
            McapTopic.ObservationCameraDown.value,
            McapTopic.ObservationCameraUp.value,
        ]

        if alignment_topic not in valid_topics:
            raise ValueError(
                f"Invalid topic to use for alignment: '{alignment_topic}'. "
                f"Valid options are: {', '.join(valid_topics)}"
            )

        alignment_topic = McapTopic(alignment_topic)
        topic_to_df = {
            McapTopic.ObservationStates: episode.state,
            McapTopic.Action: episode.action,
            McapTopic.ObservationCameraDown: episode.camera_down.data,
            McapTopic.ObservationCameraUp: episode.camera_up.data,
        }
        alignment_topic_data = topic_to_df[alignment_topic]
        fps = calculate_fps_from_timestamps(alignment_topic_data["timestamp"])
    else:
        alignment_topic, fps = find_lowest_frequency_topic(episode)

    logger.info("Using alignment topic: %s", alignment_topic.value)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(episode.joint_names),),
            "names": episode.joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(episode.joint_names),),
            "names": episode.joint_names,
        },
        "observation.images.downward": {
            "dtype": "video",
            "shape": (
                episode.camera_down.meta.height,
                episode.camera_down.meta.width,
                episode.camera_down.meta.channels,
            ),
            "names": ["height", "width", "channel"],
        },
        "observation.images.upward": {
            "dtype": "video",
            "shape": (
                episode.camera_up.meta.height,
                episode.camera_up.meta.width,
                episode.camera_up.meta.channels,
            ),
            "names": ["height", "width", "channel"],
        },
    }

    dataset_root = context.output_dir / "combined"
    dataset = LeRobotDataset.create(
        repo_id=dataset_root.name,
        fps=fps,
        features=features,
        root=dataset_root,
        robot_type="dual_arm_robot",
        image_writer_threads=8,  # LeRobot recommends 4 threads per camera with 0 processes
    )

    frame_count = 0
    for frame in generate_frames(episode, alignment_topic.value):
        dataset.add_frame(
            {
                "observation.state": frame.observation_state,
                "action": frame.action,
                "observation.images.downward": frame.observation_images_downward,
                "observation.images.upward": frame.observation_images_upward,
                "task": frame.task,
            }
        )
        frame_count += 1

    dataset.save_episode()
    logger.info("Episode 1 saved with %d frames", frame_count)

    expected_joint_names = set(episode.joint_names)
    for idx, (file, _) in enumerate(mcap_files[1:], start=2):
        logger.info(
            "Processing episode %d/%d: %s", idx, len(mcap_files), file.relative_path
        )

        if file.ingestion_status != roboto.IngestionStatus.Ingested:
            logger.warning("Skipping non-ingested file: %s", file.relative_path)
            continue

        episode = load_and_parse_mcap(file)

        # Validate that subsequent episodes have the same dims as the first
        joint_diff = expected_joint_names ^ set(episode.joint_names)
        if len(joint_diff):
            logger.error(
                "Episode %d has %d joints but expected %d. Difference: %r",
                idx,
                len(episode.joint_names),
                len(expected_joint_names),
                joint_diff,
            )
            logger.warning("Skipping episode %d due to dimension mismatch", idx)
            continue

        frame_count = 0
        for frame in generate_frames(episode, alignment_topic.value):
            frame_dict = {
                "observation.state": frame.observation_state,
                "action": frame.action,
                "observation.images.downward": frame.observation_images_downward,
                "observation.images.upward": frame.observation_images_upward,
                "task": frame.task,
            }
            dataset.add_frame(frame_dict)
            frame_count += 1

        dataset.save_episode()
        logger.info("Episode %d saved with %d frames", idx, frame_count)

    logger.info("Finalizing dataset")
    dataset.finalize()
    logger.info("MCAP to LeRobot conversion completed successfully")
