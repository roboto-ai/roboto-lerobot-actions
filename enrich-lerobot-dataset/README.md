# Enrich LeRobot Dataset With Derived Data

An example of how to enrich a LeRobot dataset with derived data. 

This action creates a new LeRobot dataset with all original features plus an additional `action_observation_difference` feature that contains the element-wise difference between the `action` and `observation.state` state vectors for each frame.

The output is a new dataset in Roboto that contains the enriched LeRobot dataset.

## Getting started

1. Setup a virtual environment specific to this project and install development dependencies, including the `roboto` SDK: `./scripts/setup.sh`
2. Build Docker image: `./scripts/build.sh`
3. Run Action image locally: `./scripts/run.sh <path-to-input-data-directory>`
4. Deploy to Roboto Platform: `./scripts/deploy.sh`

## Action configuration file

This Roboto Action is configured in `action.json`. Refer to Roboto's latest documentation for the expected structure.
