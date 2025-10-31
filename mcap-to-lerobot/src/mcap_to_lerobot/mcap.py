from __future__ import annotations

from enum import Enum
import typing

import roboto

from .extract import extract_joint_states, extract_trajectories, build_camera_data_index
from .logger import logger
from .types import EpisodeData

if typing.TYPE_CHECKING:
    import pandas as pd


NANO_SEC_PER_SEC = 1_000_000_000


class McapTopic(str, Enum):
    Action = "/arm_controller/joint_trajectory"
    ObservationStates = "/joint_states"
    ObservationCameraDown = "/face_downward/zed/left/image_rect_color/compressed"
    ObservationCameraUp = "/face_upward/zed/left/image_rect_color/compressed"


def load_and_parse_mcap(file: roboto.File) -> EpisodeData:
    """
    Load and parse MCAP file to extract all required data.

    Args:
        file: MCAP file to load
        alignment_topic: ROS topic to use as base for frame alignment
    """
    logger.info("Extracting trajectory topic")
    trajectory_topic = file.get_topic(McapTopic.Action.value)
    if not trajectory_topic:
        raise ValueError("Trajectory topic not found in MCAP file")

    trajectory_df = trajectory_topic.get_data_as_df()
    logger.info("Loaded %d trajectory messages", len(trajectory_df))

    trajectories_parsed = extract_trajectories(trajectory_df)
    trajectory_joint_names = trajectories_parsed.joint_names

    logger.info("Extracting joint states topic")
    joint_states_topic = file.get_topic(McapTopic.ObservationStates.value)
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
    downward_camera_topic = file.get_topic(McapTopic.ObservationCameraDown.value)
    upward_camera_topic = file.get_topic(McapTopic.ObservationCameraUp.value)

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

    camera_down, camera_up = build_camera_data_index(
        downward_camera_df,
        upward_camera_df,
    )

    return EpisodeData(
        action=trajectories_parsed.data,
        state=joint_states_parsed.data,
        joint_names=joint_states_parsed.joint_names,
        camera_down=camera_down,
        camera_up=camera_up,
    )


def calculate_fps_from_timestamps(timestamps: pd.Series) -> int:
    """
    Calculate FPS from timestamp array using median time difference.

    Args:
        timestamps: Array of int64 nanosecond timestamps

    Returns:
        Calculated FPS (capped at MAX_FPS)
    """
    if len(timestamps) < 2:
        raise ValueError("Need at least 2 timestamps to calculate FPS")

    time_diffs = timestamps.diff()
    median_time_diff_ns = time_diffs.median()

    # Handle edge case where all timestamps are identical or very close
    if median_time_diff_ns <= 0:
        raise ValueError(
            f"Invalid median time difference: {median_time_diff_ns}ns. "
            "All timestamps appear to be identical or not monotonically increasing."
        )

    return round(NANO_SEC_PER_SEC / median_time_diff_ns)


def find_lowest_frequency_topic(episode: EpisodeData) -> tuple[McapTopic, int]:
    """
    Find the topic with the lowest frequency (FPS) in the episode data.
    """
    topics_with_fps = []

    # Calculate FPS for each topic with better error context
    topic_data_map = [
        (McapTopic.Action, episode.action["timestamp"]),
        (McapTopic.ObservationStates, episode.state["timestamp"]),
        (McapTopic.ObservationCameraDown, episode.camera_down.data["timestamp"]),
        (McapTopic.ObservationCameraUp, episode.camera_up.data["timestamp"]),
    ]

    for topic, timestamps in topic_data_map:
        fps = calculate_fps_from_timestamps(timestamps)
        topics_with_fps.append((topic, fps))

    return min(topics_with_fps, key=lambda x: x[1])
