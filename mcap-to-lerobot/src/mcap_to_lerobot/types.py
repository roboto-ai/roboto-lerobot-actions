from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class JointStates(typing.NamedTuple):
    """Parsed joint states data."""

    joint_names: list[str]
    """List of joint names (filtered and ordered if filter_joint_names provided)"""
    data: pd.DataFrame
    """DataFrame with timestamp and joint_positions columns, sorted by timestamp"""


class Trajectories(typing.NamedTuple):
    """Parsed trajectory data with flattened trajectory points."""

    joint_names: list[str]
    """List of joint names from the trajectory"""
    data: pd.DataFrame
    """DataFrame with timestamp and trajectory_positions columns.
    Each row represents a single trajectory point with absolute timestamp."""


class ImageMeta(typing.NamedTuple):
    height: int
    width: int
    channels: int


class CameraData(typing.NamedTuple):
    """Camera data with metadata and image data for a single camera."""

    meta: ImageMeta
    data: pd.DataFrame
    """DataFrame with timestamp, format, data columns for camera images"""


class EpisodeData(typing.NamedTuple):
    """Complete episode data from MCAP file."""

    action: pd.DataFrame
    state: pd.DataFrame
    camera_down: CameraData
    camera_up: CameraData
    joint_names: list[str]


class Frame(typing.NamedTuple):
    """Single frame of observation-action data."""

    observation_state: np.ndarray
    """Joint positions at observation time"""
    action: np.ndarray
    """Target joint positions from trajectory"""
    observation_images_downward: np.ndarray
    """Downward-facing camera image (H, W, C)"""
    observation_images_upward: np.ndarray
    """Upward-facing camera image (H, W, C)"""
    task: str
    """Task identifier (constant "dual_arm_manipulation")"""
