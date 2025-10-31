from typing import Generator

import cv2
import pandas as pd

from .logger import logger
from .extract import decompress_image
from .mcap import McapTopic
from .types import EpisodeData, Frame


def generate_frames(
    episode_data: EpisodeData,
    alignment_topic: str,
) -> Generator[Frame, None, None]:
    """
    Generate observation-action pairs based on alignment_topic frequency.

    Synchronizes all signals using backward lookup based on the configured alignment_topic.

    Args:
        episode_data: Episode data containing all signals
        alignment_topic: ROS topic to use as base timeline for frame alignment
    """
    logger.info("Generating frames from episode data")

    # Prepare signal dataframes with appropriate column names
    joint_states_df = episode_data.state.rename(
        columns={"joint_positions": "observation_state"}
    )
    trajectories_df = episode_data.action.rename(
        columns={"trajectory_positions": "action"}
    )
    camera_down_df = episode_data.camera_down.data[
        ["timestamp", "format", "data"]
    ].rename(columns={"format": "format_down", "data": "camera_down"})
    camera_up_df = episode_data.camera_up.data[["timestamp", "format", "data"]].rename(
        columns={"format": "format_up", "data": "camera_up"}
    )

    # Create signal mapping
    signal_map = {
        McapTopic.ObservationStates.value: joint_states_df,
        McapTopic.Action.value: trajectories_df,
        McapTopic.ObservationCameraDown.value: camera_down_df,
        McapTopic.ObservationCameraUp.value: camera_up_df,
    }

    # Get base dataframe based on alignment_topic
    base_df = (
        signal_map[alignment_topic].sort_values("timestamp").reset_index(drop=True)
    )

    # Merge all other signals onto base_df
    merged_df = base_df
    for topic, signal_df in signal_map.items():
        if topic == alignment_topic:
            continue  # Skip the base signal

        logger.info("Merging %s onto base timeline", topic)
        merged_df = pd.merge_asof(
            merged_df,
            signal_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    # Drop rows with any NaN values
    rows_before = len(merged_df)
    merged_df = merged_df.dropna()
    rows_after = len(merged_df)
    rows_dropped = rows_before - rows_after

    logger.info(
        "Dropped %d rows with NaN values, retained %d rows",
        rows_dropped,
        rows_after,
    )

    frame_count = 0
    for _, row in merged_df.iterrows():
        observation_state = row["observation_state"]
        action = row["action"]
        image_down = decompress_image(row["camera_down"], row["format_down"])
        image_up = decompress_image(row["camera_up"], row["format_up"])

        # Resize images to match expected dimensions from camera metadata
        expected_downward_height = episode_data.camera_down.meta.height
        expected_downward_width = episode_data.camera_down.meta.width
        expected_upward_height = episode_data.camera_up.meta.height
        expected_upward_width = episode_data.camera_up.meta.width

        if image_down.shape[:2] != (
            expected_downward_height,
            expected_downward_width,
        ):
            logger.debug(
                "Resizing downward image from %s to (%d, %d, 3)",
                image_down.shape,
                expected_downward_height,
                expected_downward_width,
            )
            image_down = cv2.resize(
                image_down,
                (expected_downward_width, expected_downward_height),
                interpolation=cv2.INTER_LINEAR,
            )

        if image_up.shape[:2] != (expected_upward_height, expected_upward_width):
            logger.debug(
                "Resizing upward image from %s to (%d, %d, 3)",
                image_up.shape,
                expected_upward_height,
                expected_upward_width,
            )
            image_up = cv2.resize(
                image_up,
                (expected_upward_width, expected_upward_height),
                interpolation=cv2.INTER_LINEAR,
            )

        # Ensure action has the same shape as joint_positions
        if action.shape != observation_state.shape:
            logger.warning(
                "Action shape %s does not match joint_positions shape %s. Skipping frame.",
                action.shape,
                observation_state.shape,
            )
            continue

        frame = Frame(
            observation_state=observation_state,
            action=action,
            observation_images_downward=image_down,
            observation_images_upward=image_up,
            task="dual_arm_manipulation",
        )

        frame_count += 1
        yield frame

    logger.info("Generated %d frames", frame_count)
