import roboto

from .logger import logger


def main(context: roboto.InvocationContext) -> None:
    logger.setLevel(context.log_level)

    action_input = context.get_input()

    if action_input.files:
        logger.info("Processing %d input file(s):", len(action_input.files))
        for file, local_path in action_input.files:
            indent = "  "
            logger.info(
                "%sFile: %s (ID: %s)",
                indent,
                file.relative_path,
                file.file_id,
            )

            ### EXAMPLE: Work with file data ###
            if local_path:
                # File is downloaded and available at local_path
                logger.info("%sLocal path: %s", indent, local_path)
                logger.info("%sFile size: %d bytes", indent, local_path.stat().st_size)
            else:
                logger.warning(
                    "%sFile not downloaded (requires_downloaded_inputs may be false)",
                    indent,
                )

            ### EXAMPLE: Work with topic data ###
            # If this file has been ingested by Roboto,
            # the `file` resource provides a utility for listing the topics indexed within that file
            for topic in file.get_topics():
                logger.info(
                    "%sContains topic: %s (%s)",
                    indent,
                    topic.topic_name,
                    topic.schema_name,
                )

            # The `file` resource also provides utilities for efficiently fetching topic data
            # topic = file.get_topic("<topic_name>")  # e.g., "sensor_mag", "/rosout", etc
            # df = topic.get_data_as_df()
            # logger.info("%s%s has %d messages", indent, topic.topic_name, len(df))

    else:
        logger.info("No input files provided")
