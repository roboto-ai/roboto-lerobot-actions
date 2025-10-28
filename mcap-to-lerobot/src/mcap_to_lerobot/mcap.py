import roboto

from .extract import extract_joint_states, extract_trajectories, build_camera_data_index
from .logger import logger
from .types import EpisodeData


def load_and_parse_mcap(file: roboto.File) -> EpisodeData:
    """
    Load and parse MCAP file to extract all required data.
    """
    logger.info("Loading MCAP file: %s", file.relative_path)

    logger.info("Extracting trajectory commands topic")
    trajectory_topic = file.get_topic("/arm_controller/joint_trajectory")
    if not trajectory_topic:
        raise ValueError("Trajectory topic not found in MCAP file")

    trajectory_df = trajectory_topic.get_data_as_df()
    logger.info("Loaded %d trajectory messages", len(trajectory_df))

    trajectories_parsed = extract_trajectories(trajectory_df)
    trajectory_joint_names = trajectories_parsed.joint_names

    logger.info("Extracting joint states topic")
    joint_states_topic = file.get_topic("/joint_states")
    if not joint_states_topic:
        raise ValueError("Joint states topic not found in MCAP file")

    joint_states_df = joint_states_topic.get_data_as_df(
        message_paths_include=[
            "header",
            "name",
            "position",
        ]
    )
    logger.info("Loaded %d joint state messages", len(joint_states_df))

    if len(joint_states_df) == 0:
        raise ValueError("Joint states DataFrame is empty - no messages found in topic")

    # Filter joint states to only include joints that are in the trajectory
    joint_states_parsed = extract_joint_states(
        joint_states_df, filter_joint_names=trajectory_joint_names
    )

    logger.info("Extracting camera image topics")
    downward_camera_topic = file.get_topic(
        "/face_downward/zed/left/image_rect_color/compressed"
    )
    upward_camera_topic = file.get_topic(
        "/face_upward/zed/left/image_rect_color/compressed"
    )

    if not downward_camera_topic or not upward_camera_topic:
        raise ValueError("Camera image topics not found in MCAP file")

    downward_camera_df = downward_camera_topic.get_data_as_df(
        message_paths_include=[
            "header",
            "format",
            "data",
        ]
    )

    upward_camera_df = upward_camera_topic.get_data_as_df(
        message_paths_include=[
            "header",
            "format",
            "data",
        ]
    )

    logger.info("Loaded %d downward camera images", len(downward_camera_df))
    logger.info("Loaded %d upward camera images", len(upward_camera_df))

    camera_data = build_camera_data_index(
        downward_camera_df,
        upward_camera_df,
    )

    return EpisodeData(
        fps=joint_states_parsed.fps,
        joint_names=joint_states_parsed.joint_names,
        camera_info=camera_data.camera_info,
        joint_states_df=joint_states_parsed.data,
        trajectories_df=trajectories_parsed.data,
        camera_data=camera_data,
    )
