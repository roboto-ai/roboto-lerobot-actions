#!/usr/bin/env bash
#
# USAGE:
#   ./scripts/run.sh [OPTIONS] <lerobot_source_dataset_path>
#
# OPTIONS:
#   ./scripts/run.sh --help
#
# EXAMPLES:
#   # Basic usage - process all episodes and upload to Roboto
#   ./scripts/run.sh /path/to/lerobot/dataset
#
#   # Process only first episode with verbose output to custom output directory
#   ./scripts/run.sh -vv --episode-limit=1 -o $(pwd)/output /home/user/lerobot/dataset
#
#   # Dry run (no upload to Roboto) with debug logging
#   ./scripts/run.sh -vvv --dry-run --episode-limit=5 /path/to/dataset
#
#   # Specify Roboto source dataset for provenance tracking
#   ./scripts/run.sh --roboto-source-dataset-id=ds_abc123 /path/to/dataset

set -euo pipefail

SCRIPTS_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PACKAGE_ROOT=$(dirname "${SCRIPTS_ROOT}")

# Early exit if virtual environment does not exist
if [ ! -f "$PACKAGE_ROOT/.venv/bin/python" ]; then
    echo "Virtual environment does not exist. Please run ./scripts/setup.sh first."
    exit 1
fi

$PACKAGE_ROOT/.venv/bin/python -m src.enrich_lerobot_dataset.entrypoint "$@"
