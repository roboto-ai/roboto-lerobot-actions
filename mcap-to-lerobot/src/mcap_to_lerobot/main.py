from lerobot.datasets.lerobot_dataset import LeRobotDataset
import roboto

from .logger import logger
from .mcap import load_and_parse_mcap
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

    logger.info("Found %d MCAP files to process", len(mcap_files))

    first_file, _ = mcap_files[0]
    logger.info(
        "Processing first MCAP file to determine FPS: %s", first_file.relative_path
    )

    if first_file.ingestion_status != roboto.IngestionStatus.Ingested:
        logger.error("First file is not ingested: %s", first_file.relative_path)
        return

    first_episode_data = load_and_parse_mcap(first_file)
    fps = first_episode_data.fps
    joint_names = first_episode_data.joint_names
    camera_info = first_episode_data.camera_info

    features = {
        # State observations
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        # Actions (joint positions from trajectory)
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        # Camera observations
        # Note: Images are passed as (H, W, C) to add_frame()
        "observation.images.downward": {
            "dtype": "video",
            "shape": (
                camera_info.downward.height,
                camera_info.downward.width,
                3,
            ),
            "names": ["height", "width", "channel"],
        },
        "observation.images.upward": {
            "dtype": "video",
            "shape": (
                camera_info.upward.height,
                camera_info.upward.width,
                3,
            ),
            "names": ["height", "width", "channel"],
        },
    }

    logger.info("Creating LeRobotDataset instance")
    dataset_root = context.output_dir / "combined"
    dataset = LeRobotDataset.create(
        repo_id=dataset_root.name,
        fps=fps,
        features=features,
        root=dataset_root,
        robot_type="dual_arm_robot",
        image_writer_threads=8,  # LeRobot recommends 4 threads per camera with 0 processes
    )

    logger.info("LeRobotDataset created successfully")

    logger.info(
        "Processing episode 1/%d: %s", len(mcap_files), first_file.relative_path
    )

    frame_count = 0
    for frame in generate_frames(first_episode_data):
        # Convert Frame NamedTuple to dict format expected by LeRobotDataset
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
    logger.info("Episode 1 saved with %d frames", frame_count)

    for idx, (file, _) in enumerate(mcap_files[1:], start=2):
        logger.info(
            "Processing episode %d/%d: %s", idx, len(mcap_files), file.relative_path
        )

        if file.ingestion_status != roboto.IngestionStatus.Ingested:
            logger.warning("Skipping non-ingested file: %s", file.relative_path)
            continue

        episode_data = load_and_parse_mcap(file)

        # Validate that subsequent episodes have the same joint count as the first
        episode_joint_names = episode_data.joint_names
        if len(episode_joint_names) != len(joint_names):
            logger.error(
                "Episode %d has %d joints but expected %d. Joint names: %r vs %r",
                idx,
                len(episode_joint_names),
                len(joint_names),
                episode_joint_names,
                joint_names,
            )
            logger.warning("Skipping episode %d due to joint count mismatch", idx)
            continue

        frame_count = 0
        for frame in generate_frames(episode_data):
            # Convert Frame NamedTuple to dict format expected by LeRobotDataset
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
