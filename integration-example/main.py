import argparse
import logging
import pathlib
import tempfile

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import roboto

logging.basicConfig(
    format="[%(levelname)4s:%(filename)s %(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(name="roboto-lerobot-example")


def main(
    dataset_id: str,
    output_dir: pathlib.Path,
    dataset_path: pathlib.Path | None = None,
) -> LeRobotDataset:
    dataset = roboto.Dataset.from_id(dataset_id)
    download_kwargs = {}
    if dataset_path is not None:
        download_kwargs["include_patterns"] = [f"{dataset_path}/*"]

    log.info(
        "Downloading LeRobot dataset stored in Roboto dataset %s to %s",
        dataset_id,
        output_dir,
    )
    dataset.download_files(output_dir, **download_kwargs)

    repo_id = str(dataset_path) if dataset_path is not None else output_dir.name
    root = output_dir / repo_id if dataset_path is not None else output_dir
    return LeRobotDataset(repo_id=repo_id, root=root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_id",
        help=(
            "The ID of the Roboto dataset from which to pull LeRobot data."
        ),
        type=str
    )
    parser.add_argument(
        "-o",
        "--output-dir", 
        help=(
            "(Optional) Specify where data stored in Roboto should be downloaded locally. "
            "If not specified, a temporary directory is used. "
            "Example: --output-dir=\"$(pwd)/.cache\""
        ),
        type=pathlib.Path, 
        default=pathlib.Path(tempfile.mkdtemp())
    )
    parser.add_argument(
        "-p",
        "--path", 
        help=(
            "(Optional) Path relative to the root of the Roboto dataset that points to the directory containing LeRobot data. "
            "If not specified, it is assumed that the LeRobot data is located at the root of the Roboto dataset. "
            "Example: --path=\"recordings/2025-10-06\""
        ),
        type=pathlib.Path, 
        required=False
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    try:
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        lerobot_ds = main(args.dataset_id, args.output_dir, args.path)
        log.info("Successfully pulled data from Roboto storage into a LeRobot dataset instance")
        log.info("This LeRobot dataset contains %d episodes", lerobot_ds.num_episodes)
    except KeyboardInterrupt:
        pass
