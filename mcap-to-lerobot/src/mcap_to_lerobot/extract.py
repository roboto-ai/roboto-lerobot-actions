import cv2
import numpy as np
import pandas as pd

from .logger import logger
from .types import (
    JointStates,
    Trajectories,
    CameraData,
    CameraInfoData,
    CameraInfoDimensions,
    TrajectoryPoint,
)

NANO_SEC_PER_SEC = 1_000_000_000


def ros_time_to_nanoseconds(sec: int, nsec: int) -> int:
    """
    Convert ROS timestamp (sec + nsec) to nanoseconds since Unix epoch.

    Args:
        sec: Seconds component of ROS timestamp
        nsec: Nanoseconds component of ROS timestamp

    Returns:
        int64 nanoseconds since Unix epoch
    """
    return int(sec * NANO_SEC_PER_SEC + nsec)


def extract_joint_states(
    joint_states_df: pd.DataFrame, filter_joint_names: list[str] | None = None
) -> JointStates:
    """
    Parse joint states DataFrame to extract timestamps, joint names, and positions.
    Calculates sampling frequency from median time difference.
    Validates joint name consistency across messages.

    Args:
        joint_states_df: DataFrame containing joint state messages
        filter_joint_names: Optional list of joint names to filter and reorder output.
                          Raises ValueError if any specified joint is not found.

    Returns:
        ParsedJointStates containing:
        - fps: Calculated sampling frequency (capped at 240 for video encoder compatibility)
        - joint_names: List of joint names (filtered and ordered if filter_joint_names provided)
        - dataframe: DataFrame with timestamp and joint_positions columns, sorted by timestamp
    """
    timestamps = []
    joint_names_list = []
    positions_list = []

    for _, row in joint_states_df.iterrows():
        timestamp_ns = ros_time_to_nanoseconds(
            row["header.stamp.sec"], row["header.stamp.nsec"]
        )
        timestamps.append(timestamp_ns)

        joint_names = row["name"]
        joint_names_list.append(joint_names)

        positions = np.array(row["position"], dtype=np.float32)

        # Filter positions if filter_joint_names is provided
        if filter_joint_names is not None:
            # Create a mapping from joint name to position
            joint_name_to_position = dict(zip(joint_names, positions))

            # Extract positions in the order specified by filter_joint_names
            filtered_positions = []
            for joint_name in filter_joint_names:
                if joint_name not in joint_name_to_position:
                    raise ValueError(
                        f"Joint '{joint_name}' from trajectory not found in joint states. "
                        f"Available joints: {list(joint_names)}"
                    )
                filtered_positions.append(joint_name_to_position[joint_name])

            positions = np.array(filtered_positions, dtype=np.float32)

        positions_list.append(positions)

    # Validate that all messages have the same joint names
    first_joint_names = joint_names_list[0]
    for idx, names in enumerate(joint_names_list[1:], start=1):
        if list(names) != list(first_joint_names):
            logger.warning("Joint names mismatch at message %d", idx)

    # Determine which joint names to return
    if filter_joint_names is not None:
        output_joint_names = filter_joint_names
    else:
        output_joint_names = list(first_joint_names)

    parsed_df = pd.DataFrame(
        {"timestamp": timestamps, "joint_positions": positions_list}
    )

    parsed_df = parsed_df.sort_values("timestamp").reset_index(drop=True)

    time_diffs = np.diff(parsed_df["timestamp"].values)
    median_time_diff_ns = np.median(time_diffs)
    calculated_fps = round(NANO_SEC_PER_SEC / median_time_diff_ns)

    # Cap FPS at 240 to comply with SVT-AV1 encoder maximum
    MAX_FPS = 240
    if calculated_fps > MAX_FPS:
        logger.warning(
            "Calculated FPS (%d) exceeds maximum supported by video encoder (%d). Capping at %d fps.",
            calculated_fps,
            MAX_FPS,
            MAX_FPS,
        )
        fps = MAX_FPS
    else:
        fps = calculated_fps

    return JointStates(fps=fps, joint_names=output_joint_names, data=parsed_df)


def extract_trajectories(trajectory_df: pd.DataFrame) -> Trajectories:
    """
    Parse trajectory DataFrame to extract timestamps, trajectory points, and joint names.
    """
    timestamps = []
    points_list = []
    joint_names_list = []

    for _, row in trajectory_df.iterrows():
        timestamp_ns = ros_time_to_nanoseconds(
            row["header.stamp.sec"], row["header.stamp.nsec"]
        )
        timestamps.append(timestamp_ns)

        if not joint_names_list:
            joint_names_list = row["joint_names"]

        parsed_points = []
        for point in row["points"]:
            positions = np.array(point["positions"], dtype=np.float32)
            time_from_start = point["time_from_start"]
            time_from_start_ns = ros_time_to_nanoseconds(
                time_from_start["sec"], time_from_start["nsec"]
            )
            parsed_points.append(
                TrajectoryPoint(
                    positions=positions, time_from_start_ns=time_from_start_ns
                )
            )

        points_list.append(parsed_points)
        if not parsed_points:
            logger.warning(
                "Trajectory message at timestamp %d has no valid points.", timestamp_ns
            )

    if not joint_names_list:
        raise ValueError("No joint names found in trajectory messages")

    parsed_df = pd.DataFrame({"timestamp": timestamps, "points": points_list})
    parsed_df = parsed_df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Parsed %d trajectory messages with valid points", len(parsed_df))

    return Trajectories(
        data=parsed_df,
        joint_names=list(joint_names_list),
    )


def build_camera_data_index(
    downward_camera_df: pd.DataFrame,
    upward_camera_df: pd.DataFrame,
    downward_info_df: pd.DataFrame,
    upward_info_df: pd.DataFrame,
) -> CameraData:
    """
    Build camera data index by merging image data with CameraInfo.
    """
    logger.info("Building camera data index")

    # Downward camera images
    downward_timestamps = []
    downward_data = []

    for _, row in downward_camera_df.iterrows():
        timestamp_ns = ros_time_to_nanoseconds(
            row["header.stamp.sec"], row["header.stamp.nsec"]
        )
        downward_timestamps.append(timestamp_ns)
        downward_data.append(
            {
                "timestamp": timestamp_ns,
                "format": row["format"],
                "data": row["data"],
            }
        )

    downward_df = pd.DataFrame(downward_data)

    # Upward camera images
    upward_timestamps = []
    upward_data = []

    for _, row in upward_camera_df.iterrows():
        timestamp_ns = ros_time_to_nanoseconds(
            row["header.stamp.sec"], row["header.stamp.nsec"]
        )
        upward_timestamps.append(timestamp_ns)
        upward_data.append(
            {
                "timestamp": timestamp_ns,
                "format": row["format"],
                "data": row["data"],
            }
        )

    upward_df = pd.DataFrame(upward_data)

    # Downward camera info
    downward_info_timestamps = []
    downward_info_data = []

    for _, row in downward_info_df.iterrows():
        timestamp_ns = ros_time_to_nanoseconds(
            row["header.stamp.sec"], row["header.stamp.nsec"]
        )
        downward_info_timestamps.append(timestamp_ns)
        downward_info_data.append(
            {
                "timestamp": timestamp_ns,
                "height": int(row["height"]),
                "width": int(row["width"]),
            }
        )

    downward_info_parsed = pd.DataFrame(downward_info_data)

    # Upward camera info
    upward_info_timestamps = []
    upward_info_data = []

    for _, row in upward_info_df.iterrows():
        timestamp_ns = ros_time_to_nanoseconds(
            row["header.stamp.sec"], row["header.stamp.nsec"]
        )
        upward_info_timestamps.append(timestamp_ns)
        upward_info_data.append(
            {
                "timestamp": timestamp_ns,
                "height": int(row["height"]),
                "width": int(row["width"]),
            }
        )

    upward_info_parsed = pd.DataFrame(upward_info_data)

    # Merge camera info with images
    downward_merged = pd.merge_asof(
        downward_df.sort_values("timestamp"),
        downward_info_parsed.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        suffixes=("", "_info"),
    )

    upward_merged = pd.merge_asof(
        upward_df.sort_values("timestamp"),
        upward_info_parsed.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        suffixes=("", "_info"),
    )

    downward_heights = downward_merged["height"].dropna().unique()
    downward_widths = downward_merged["width"].dropna().unique()
    upward_heights = upward_merged["height"].dropna().unique()
    upward_widths = upward_merged["width"].dropna().unique()

    if len(downward_heights) > 1 or len(downward_widths) > 1:
        logger.warning(
            "Inconsistent downward camera dimensions: heights=%r, widths=%r",
            downward_heights,
            downward_widths,
        )

    if len(upward_heights) > 1 or len(upward_widths) > 1:
        logger.warning(
            "Inconsistent upward camera dimensions: heights=%r, widths=%r",
            upward_heights,
            upward_widths,
        )

    # Get final dimensions
    downward_height = int(downward_merged["height"][0])
    downward_width = int(downward_merged["width"][0])
    upward_height = int(upward_merged["height"][0])
    upward_width = int(upward_merged["width"][0])

    return CameraData(
        camera_info=CameraInfoData(
            downward=CameraInfoDimensions(height=downward_height, width=downward_width),
            upward=CameraInfoDimensions(height=upward_height, width=upward_width),
        ),
        downward=downward_merged,
        upward=upward_merged,
    )


def decompress_image(compressed_data: bytes, format_str: str) -> np.ndarray:
    """
    Decompress CompressedImage data using cv2.imdecode and convert BGR to RGB.

    Args:
        compressed_data: Compressed image bytes from ROS CompressedImage message
        format_str: Format string from CompressedImage message (e.g., "jpeg", "png")

    Returns:
        numpy array (H, W, 3) uint8 in RGB format
    """
    img_array = np.frombuffer(compressed_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image with format: %s" % format_str)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb


def get_trajectory_action_at_time(
    observation_timestamp: int,
    trajectory_timestamp: int,
    trajectory_points: list[TrajectoryPoint],
) -> np.ndarray:
    """
    Get action from trajectory points based on elapsed time.

    Args:
        observation_timestamp: Current observation timestamp (int64 nanoseconds)
        trajectory_timestamp: Trajectory start timestamp (int64 nanoseconds)
        trajectory_points: List of TrajectoryPoint with positions and time_from_start_ns

    Returns:
        numpy array of target joint positions (float32)
    """
    elapsed_time_ns = observation_timestamp - trajectory_timestamp

    # Handle edge cases
    if elapsed_time_ns < trajectory_points[0].time_from_start_ns:
        return trajectory_points[0].positions

    if elapsed_time_ns >= trajectory_points[-1].time_from_start_ns:
        return trajectory_points[-1].positions

    # Find the appropriate trajectory point (no interpolation between points)
    for i in range(len(trajectory_points) - 1, -1, -1):
        if trajectory_points[i].time_from_start_ns <= elapsed_time_ns:
            return trajectory_points[i].positions

    # Fallback (should not reach here)
    return trajectory_points[0].positions
