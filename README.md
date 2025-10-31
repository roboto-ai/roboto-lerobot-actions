# Roboto LeRobot Actions

This repository contains example Roboto Actions demonstrating the interoperability between [LeRobot datasets](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) and the [Roboto](https://www.roboto.ai/) platform.

## Overview

LeRobot is an open-source framework for robotics machine learning that defines its own dataset format. This repository provides **example implementations** showing how you can integrate LeRobot datasets with Roboto's data management and processing platform through custom Actions. These examples demonstrate patterns for loading, processing, and enriching LeRobot datasets stored in Roboto using the Actions Platform.

## Actions

### Convert MCAP Files to LeRobot Dataset

**An example Roboto Action** that demonstrates how to convert MCAP files stored in Roboto into LeRobot dataset format. This example shows how to:
- Load and parse MCAP files from Roboto storage
- Extract and synchronize multiple topics (joint states, trajectories, camera images)
- Generate LeRobot-compatible datasets with proper metadata and structure

In this specific example, the action processes MCAP files where each file represents one episode of dual-arm robot manipulation. It extracts action trajectories, observation states, and dual camera feeds, then aligns them temporally based on a configurable topic frequency. The resulting LeRobot dataset includes synchronized observation-action pairs with video data, ready for model training.

See the [action-specific README](./mcap-to-lerobot/README.md) for detailed setup and usage instructions.

### Enrich LeRobot Dataset With Derived Data

**An example Roboto Action** that demonstrates one approach to adding features to LeRobot datasets stored in Roboto. This example shows how to:
- Load LeRobot datasets from Roboto storage
- Process and compute derived features from existing dataset data
- Use LeRobot's native `add_features` API to create enriched datasets
- Upload the enriched results back to Roboto as new datasets

In this specific example, the action adds an `action_observation_difference` feature to LeRobot datasets by computing the element-wise difference between action and observation state vectors for each frame. This pattern can be adapted to add any custom features or transformations to your LeRobot datasets.

See the [action-specific README](./enrich-lerobot-dataset/README.md) for detailed setup and usage instructions.

## Resources

- [Roboto Platform Documentation](https://docs.roboto.ai/)
- [Roboto Actions Guide](https://docs.roboto.ai/user-guides/process-data-actions.html)
- [LeRobot Dataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
