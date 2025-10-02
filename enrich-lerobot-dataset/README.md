# Enrich LeRobot Dataset With Derived Data

An example of how to enrich a LeRobot dataset with derived data.

This action creates a new LeRobot dataset with all original features plus an additional `action_observation_difference` feature that contains the element-wise difference between the `action` and `observation.state` state vectors for each frame.

The output is a new dataset in Roboto that contains the enriched LeRobot dataset.

## Local usage

1. `./scripts/setup.sh`: setup a virtual environment specific to this project and install dependencies, including the `roboto` SDK
2. `./scripts/run.sh <local-path-to-lerobot-dataset>`: see `--help` for usage.

## Deployment to Roboto

1. `./scripts/build.sh`: build as Docker image
2. `./scripts/deploy.sh`: deploy to Roboto Platform

## Action configuration file

This Roboto Action is configured in `action.json`. Refer to Roboto's latest documentation for the expected structure.
