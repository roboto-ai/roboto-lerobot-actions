import multiprocessing
import shutil
import pathlib

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_from_directory(directory: pathlib.Path):
    return LeRobotDataset(repo_id=directory.parent.name, root=directory)


def clone(
    source: LeRobotDataset,
    outdir: pathlib.Path,
    features: dict[str, dict] | None = None,
) -> LeRobotDataset:
    features = features if features is not None else source.meta.features
    repo_id = "clone"
    root = outdir / repo_id

    if root.exists():
        shutil.rmtree(root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        features=features,
        fps=source.fps,
        robot_type=source.meta.robot_type,
        image_writer_threads=multiprocessing.cpu_count() * 2,  # assume hyperthreading
    )
