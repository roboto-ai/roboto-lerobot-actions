import argparse
import logging
import pathlib

from .images import find_reorder_permutation
from .lerobot_dataset import load_from_directory, clone
from .roboto_dataset import create_roboto_dataset

logging.basicConfig(
    format="[%(levelname)4s:%(filename)s %(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(name="enrich_lerobot_dataset")


class Args(argparse.Namespace):
    dry_run: bool = False
    episode_limit: int | None = None
    lerobot_source_dataset_path: pathlib.Path  # Path to LeRobot dataset
    output_dir: pathlib.Path
    roboto_source_dataset_id: str | None  # Roboto Dataset ID
    roboto_org_id: str | None
    log_level: int  # logging.ERROR | logging.INFO | logging.DEBUG


def main(args: Args):
    source_lerobot_ds = load_from_directory(args.lerobot_source_dataset_path)
    source_features = source_lerobot_ds.meta.features
    action_feature = source_features["action"]

    ACTION_OBSERVATION_DIFFERENCE = "action_observation_difference"
    enriched_features = {
        **source_features,
        ACTION_OBSERVATION_DIFFERENCE: {
            "dtype": action_feature["dtype"],
            "shape": action_feature["shape"],
            "names": action_feature["names"],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    enriched_lerobot_ds = clone(
        source_lerobot_ds, args.output_dir, features=enriched_features
    )

    episode_limit = int(args.episode_limit) if args.episode_limit is not None else None
    episodes_to_process = (
        episode_limit if episode_limit is not None else source_lerobot_ds.num_episodes
    )
    for episode_idx in range(episodes_to_process):
        log.info(
            "---- Begin processing episode %d / %d ----",
            episode_idx + 1,
            episodes_to_process,
        )

        start_idx = int(
            source_lerobot_ds.episode_data_index["from"][episode_idx].item()
        )
        end_idx = int(source_lerobot_ds.episode_data_index["to"][episode_idx].item())

        for frame_idx in range(start_idx, end_idx):
            frame_data = source_lerobot_ds[frame_idx]

            task = frame_data.pop("task")
            timestamp = frame_data.pop("timestamp")
            for field in ["frame_index", "episode_index", "index", "task_index"]:
                frame_data.pop(field)

            for field in frame_data.keys():
                feature = enriched_features[field]
                if feature["dtype"] == "video":
                    expected_shape = feature["shape"]
                    tensor_shape = tuple(frame_data[field].shape)
                    is_same_order, permutation = find_reorder_permutation(
                        expected_shape, tensor_shape
                    )
                    if not is_same_order:
                        frame_data[field] = frame_data[field].permute(*permutation)

            # enrich with difference between action and observation.state
            diff = frame_data["action"] - frame_data["observation.state"]
            frame_data[ACTION_OBSERVATION_DIFFERENCE] = diff

            enriched_lerobot_ds.add_frame(frame_data, task, timestamp.item())

        enriched_lerobot_ds.save_episode()
        log.info(
            "---- End processing episode %d / %d ----",
            episode_idx + 1,
            episodes_to_process,
        )

    log.info("Finished processing episodes.")

    if args.dry_run:
        log.info("Dry run enabled, skipping upload to Roboto.")
        return

    roboto_ds = create_roboto_dataset(
        derived_from=args.roboto_source_dataset_id, org_id=args.roboto_org_id
    )

    log.info("Uploading enriched dataset to Roboto: %s", roboto_ds.dataset_id)
    roboto_ds.upload_directory(args.output_dir / enriched_lerobot_ds.repo_id)

    log.info(
        "Upload of enriched LeRobot dataset to Roboto datasets %s complete",
        roboto_ds.dataset_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "lerobot_source_dataset_path",
        type=pathlib.Path,
        help="Local filesystem path to LeRobot dataset.",
    )

    parser.add_argument(
        "-l",
        "--episode-limit",
        type=int,
        default=None,
        help="Limit the number of episodes to process.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent.parent / "output",
        help="Local filesystem path to output directory.",
    )

    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode, i.e. do not upload to Roboto.",
    )

    parser.add_argument(
        "--roboto-source-dataset-id",
        type=str,
        default=None,
        help=(
            "Roboto Dataset ID. "
            "If provided, the enriched dataset will note that it was derived from this dataset in its description."
        ),
    )

    parser.add_argument(
        "--roboto-org-id",
        type=str,
        default=None,
        help=(
            "Roboto Organization ID. "
            "Set if you are a member of multiple Roboto orgs. Unlikely to be required by most users."
        ),
    )

    class VerbosityAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            kwargs.setdefault("default", logging.ERROR)
            kwargs.setdefault("nargs", 0)
            super().__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            current_level = getattr(namespace, self.dest, logging.ERROR)
            new_level = max(current_level - 10, logging.DEBUG)
            setattr(namespace, self.dest, new_level)

    parser.add_argument(
        "-v",
        "--verbose",
        action=VerbosityAction,
        dest="log_level",
        help=(
            "Set increasing levels of verbosity. "
            "Only error logs are printed by default. "
            "Use -v (warn), -vv (info), -vvv (debug)."
        ),
    )
    args = parser.parse_args(namespace=Args())

    log.setLevel(args.log_level)

    main(args)
