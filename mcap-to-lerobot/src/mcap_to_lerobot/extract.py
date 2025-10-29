import cv2
import numpy as np
import pandas as pd

from .logger import logger
from .types import (
    JointStates,
    Trajectories,
    CameraData,
    ImageMeta,
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
    Validates joint name consistency across messages.

    Args:
        joint_states_df: DataFrame containing joint state messages
        filter_joint_names: Optional list of joint names to filter and reorder output.
                          Raises ValueError if any specified joint is not found.

    Returns:
        ParsedJointStates containing:
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

    parsed_df = (
        pd.DataFrame({"timestamp": timestamps, "joint_positions": positions_list})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return JointStates(joint_names=output_joint_names, data=parsed_df)


def keep_monotonic_timestamps(df: pd.DataFrame):
    """
    Keep only rows where timestamps are monotonically increasing.
    When a timestamp decreases, pop previous rows until finding one less than the current value.

    Args:
        df: DataFrame with a 'timestamp' column

    Returns:
        DataFrame with only monotonically increasing timestamps
    """
    keep = []
    current_max = float("-inf")

    for i, t in enumerate(df["timestamp"]):
        if t >= current_max:
            # still (or newly) monotonic
            keep.append(i)
            current_max = t
        else:
            # rewind: pop indices until t > previous element
            while keep and df["timestamp"].iloc[keep[-1]] > t:
                keep.pop()
            keep.append(i)
            current_max = t

    return df.iloc[keep].reset_index(drop=True)


def extract_trajectories(trajectory_df: pd.DataFrame) -> Trajectories:
    """
    Parse trajectory DataFrame to extract flattened trajectory points with absolute timestamps.

    Each trajectory message contains multiple points with relative time_from_start.
    This function flattens all points into individual rows with absolute timestamps.

    Returns:
        Trajectories with data DataFrame containing:
        - timestamp: absolute timestamp (int64 nanoseconds since Unix epoch)
        - trajectory_positions: joint positions array (np.ndarray of float32)
    """
    timestamps = []
    positions_list = []
    joint_names_list = []

    for _, row in trajectory_df.iterrows():
        trajectory_timestamp_ns = ros_time_to_nanoseconds(
            row["header.stamp.sec"], row["header.stamp.nsec"]
        )

        if not joint_names_list:
            joint_names_list = row["joint_names"]

        # Flatten trajectory points into individual rows with absolute timestamps
        for point in row["points"]:
            positions = np.array(point["positions"], dtype=np.float32)
            time_from_start = point["time_from_start"]
            time_from_start_ns = ros_time_to_nanoseconds(
                time_from_start["sec"], time_from_start["nsec"]
            )

            # Calculate absolute timestamp for this trajectory point
            absolute_timestamp = trajectory_timestamp_ns + time_from_start_ns

            timestamps.append(absolute_timestamp)
            positions_list.append(positions)

        if not row["points"]:
            logger.warning(
                "Trajectory message at timestamp %d has no valid points.",
                trajectory_timestamp_ns,
            )

    if not joint_names_list:
        raise ValueError("No joint names found in trajectory messages")

    parsed_df = pd.DataFrame(
        {"timestamp": timestamps, "trajectory_positions": positions_list}
    )
    parsed_df = keep_monotonic_timestamps(parsed_df)
    logger.info(
        "Parsed %d flattened trajectory points from trajectory messages", len(parsed_df)
    )

    return Trajectories(
        data=parsed_df,
        joint_names=list(joint_names_list),
    )


def build_camera_data_index(
    downward_camera_df: pd.DataFrame,
    upward_camera_df: pd.DataFrame,
) -> tuple[CameraData, CameraData]:
    """
    Build camera data index and detect image dimensions from first image of each camera.

    Returns:
        Tuple of (camera_down, camera_up) CameraData objects
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

    # Detect dimensions from first image of each camera
    if len(downward_df) == 0:
        raise ValueError("No downward camera images found")
    if len(upward_df) == 0:
        raise ValueError("No upward camera images found")

    logger.info("Detecting image dimensions from first frame of each camera")

    first_downward = downward_df.iloc[0]
    downward_height, downward_width, downward_channels = get_image_dimensions(
        first_downward["data"], first_downward["format"]
    )
    logger.info(
        "Downward camera dimensions: %dx%dx%d",
        downward_height,
        downward_width,
        downward_channels,
    )

    first_upward = upward_df.iloc[0]
    upward_height, upward_width, upward_channels = get_image_dimensions(
        first_upward["data"], first_upward["format"]
    )
    logger.info(
        "Upward camera dimensions: %dx%dx%d",
        upward_height,
        upward_width,
        upward_channels,
    )

    camera_down = CameraData(
        meta=ImageMeta(
            height=downward_height, width=downward_width, channels=downward_channels
        ),
        data=downward_df,
    )

    camera_up = CameraData(
        meta=ImageMeta(
            height=upward_height, width=upward_width, channels=upward_channels
        ),
        data=upward_df,
    )

    return camera_down, camera_up


def get_image_dimensions(
    compressed_data: bytes, format_str: str
) -> tuple[int, int, int]:
    """
    Get image dimensions without full decompression.

    Args:
        compressed_data: Compressed image bytes from ROS CompressedImage message
        format_str: Format string from CompressedImage message (e.g., "jpeg", "png")

    Returns:
        Tuple of (height, width, channels) in pixels
    """
    img_array = np.frombuffer(compressed_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image with format: %s" % format_str)

    return img.shape


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
