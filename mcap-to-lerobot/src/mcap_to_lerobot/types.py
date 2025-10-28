from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class JointStates(typing.NamedTuple):
    """Parsed joint states data."""

    fps: int
    """Calculated sampling frequency"""
    joint_names: list[str]
    """List of joint names (filtered and ordered if filter_joint_names provided)"""
    data: pd.DataFrame
    """DataFrame with timestamp and joint_positions columns, sorted by timestamp"""


class Trajectories(typing.NamedTuple):
    """Parsed trajectory data."""

    joint_names: list[str]
    """List of joint names from the trajectory"""
    data: pd.DataFrame
    """DataFrame with timestamp and points columns"""


class CameraInfoDimensions(typing.NamedTuple):
    """Camera dimensions."""

    height: int
    """Image height in pixels"""
    width: int
    """Image width in pixels"""


class CameraInfoData(typing.NamedTuple):
    """Camera information for both cameras."""

    downward: CameraInfoDimensions
    """Dimensions for downward-facing camera"""
    upward: CameraInfoDimensions
    """Dimensions for upward-facing camera"""


class CameraData(typing.NamedTuple):
    """Camera data index with merged image and info data."""

    camera_info: CameraInfoData
    """Camera dimensions for both cameras"""
    downward: pd.DataFrame
    """Merged DataFrame with timestamp, format, data, height, width columns"""
    upward: pd.DataFrame
    """Merged DataFrame with timestamp, format, data, height, width columns"""


class EpisodeData(typing.NamedTuple):
    """Complete episode data from MCAP file."""

    fps: int
    """Calculated sampling frequency from joint states"""
    joint_names: list[str]
    """List of joint names (filtered to match trajectory joints)"""
    camera_info: CameraInfoData
    """Camera dimensions for both cameras"""
    joint_states_df: pd.DataFrame
    """DataFrame with timestamp and joint_positions columns"""
    trajectories_df: pd.DataFrame
    """DataFrame with timestamp and trajectory points"""
    camera_data: CameraData
    """Camera data index with merged DataFrames"""


class Frame(typing.NamedTuple):
    """Single frame of observation-action data."""

    observation_state: np.ndarray
    """Joint positions at observation time"""
    action: np.ndarray
    """Target joint positions from trajectory"""
    observation_images_downward: np.ndarray
    """Downward-facing camera image (H, W, 3) RGB"""
    observation_images_upward: np.ndarray
    """Upward-facing camera image (H, W, 3) RGB"""
    task: str
    """Task identifier (constant "dual_arm_manipulation")"""


class TrajectoryPoint(typing.NamedTuple):
    """Single point in a trajectory."""

    positions: np.ndarray
    """Joint positions at this trajectory point"""
    time_from_start_ns: int
    """Time from trajectory start in nanoseconds"""
