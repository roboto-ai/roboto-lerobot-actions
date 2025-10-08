# Initialize a LeRobot Dataset with Roboto

This script demonstrates how to initialize a LeRobot dataset with data uploaded to Roboto.

## Setup

This script uses [uv](https://docs.astral.sh/uv/) as a package and project manager. Install uv [here](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

## Usage

```bash
$ uv run main.py --help
usage: main.py [-h] [-o OUTPUT_DIR] [-p PATH] [-v] dataset_id

positional arguments:
  dataset_id            The ID of the Roboto dataset from which to pull LeRobot data.

options:
  -h, --help            Show this help message and exit

  -o, --output-dir OUTPUT_DIR
                        (Optional) Specify where data stored in Roboto should be downloaded locally.
                        If not specified, a temporary directory is used. Example: --output-dir="$(pwd)/.cache"

  -p, --path PATH       (Optional) Path relative to the root of the Roboto dataset that points to the directory containing LeRobot data. 
                        If not specified, it is assumed that the LeRobot data is located at the root of the Roboto dataset. 
                        Example: --path="recordings/2025-10-06"
                        
  -v, --verbose
```