from typing import Generator

import cv2
import pandas as pd

from .logger import logger
from .extract import decompress_image, get_trajectory_action_at_time
from .types import EpisodeData, Frame, TrajectoryPoint


def generate_frames(
    episode_data: EpisodeData,
) -> Generator[Frame, None, None]:
    """
    Generate observation-action pairs at joint state frequency.

    Synchronizes joint states with trajectories (backward lookup) and camera images (backward lookup).
    """
    logger.info("Generating frames from episode data")

    joint_states_df = episode_data.joint_states_df
    trajectories_df = episode_data.trajectories_df
    camera_data = episode_data.camera_data

    logger.info("Merging in trajectory data")
    # Rename trajectory timestamp to avoid collision with joint state timestamp
    trajectories_df_renamed = trajectories_df.rename(
        columns={"timestamp": "timestamp_traj"}
    )

    merged_df = pd.merge_asof(
        joint_states_df.sort_values("timestamp"),
        trajectories_df_renamed.sort_values("timestamp_traj"),
        left_on="timestamp",
        right_on="timestamp_traj",
        direction="backward",
    )

    logger.info("Merged %d joint states with trajectories", len(merged_df))

    logger.info("Merging in camera frames")
    downward_camera_df = camera_data.downward.sort_values("timestamp")
    upward_camera_df = camera_data.upward.sort_values("timestamp")

    # Rename columns before merge to avoid suffix confusion
    downward_camera_df_renamed = downward_camera_df[["timestamp", "format", "data"]].rename(
        columns={"format": "format_downward", "data": "data_downward"}
    )
    upward_camera_df_renamed = upward_camera_df[["timestamp", "format", "data"]].rename(
        columns={"format": "format_upward", "data": "data_upward"}
    )

    merged_df = pd.merge_asof(
        merged_df,
        downward_camera_df_renamed,
        on="timestamp",
        direction="backward",
    )

    merged_df = pd.merge_asof(
        merged_df,
        upward_camera_df_renamed,
        on="timestamp",
        direction="backward",
    )

    logger.info("Merged camera data, total frames: %d", len(merged_df))

    frame_count = 0
    for _, row in merged_df.iterrows():
        obs_timestamp = row["timestamp"]
        joint_positions = row["joint_positions"]
        traj_timestamp = row["timestamp_traj"]
        traj_points: list[TrajectoryPoint] = row["points"]

        if len(traj_points) > 0:
            action = get_trajectory_action_at_time(
                obs_timestamp, traj_timestamp, traj_points
            )
        else:
            # No trajectory available, skip
            continue

        # Check if camera data is available (not NaN from merge_asof)
        if pd.isna(row["data_downward"]) or pd.isna(row["data_upward"]):
            logger.debug("Skipping frame at timestamp %d: missing camera data", obs_timestamp)
            continue

        downward_image = decompress_image(row["data_downward"], row["format_downward"])
        upward_image = decompress_image(row["data_upward"], row["format_upward"])

        # Resize images to match expected dimensions from camera_info
        expected_downward_height = camera_data.camera_info.downward.height
        expected_downward_width = camera_data.camera_info.downward.width
        expected_upward_height = camera_data.camera_info.upward.height
        expected_upward_width = camera_data.camera_info.upward.width

        if downward_image.shape[:2] != (
            expected_downward_height,
            expected_downward_width,
        ):
            logger.debug(
                "Resizing downward image from %s to (%d, %d, 3)",
                downward_image.shape,
                expected_downward_height,
                expected_downward_width,
            )
            downward_image = cv2.resize(
                downward_image,
                (expected_downward_width, expected_downward_height),
                interpolation=cv2.INTER_LINEAR,
            )

        if upward_image.shape[:2] != (expected_upward_height, expected_upward_width):
            logger.debug(
                "Resizing upward image from %s to (%d, %d, 3)",
                upward_image.shape,
                expected_upward_height,
                expected_upward_width,
            )
            upward_image = cv2.resize(
                upward_image,
                (expected_upward_width, expected_upward_height),
                interpolation=cv2.INTER_LINEAR,
            )

        # Ensure action has the same shape as joint_positions
        if action.shape != joint_positions.shape:
            logger.warning(
                "Action shape %s does not match joint_positions shape %s. Skipping frame.",
                action.shape,
                joint_positions.shape,
            )
            continue

        frame = Frame(
            observation_state=joint_positions,
            action=action,
            observation_images_downward=downward_image,
            observation_images_upward=upward_image,
            task="dual_arm_manipulation",
        )

        frame_count += 1
        yield frame

    logger.info("Generated %d frames", frame_count)
