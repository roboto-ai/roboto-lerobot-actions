# Roboto LeRobot Actions

This repository contains Roboto Actions and example code demonstrating the interoperability between [LeRobot datasets](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) and the [Roboto](https://www.roboto.ai/) platform.

## Overview

LeRobot is an open-source framework for robotics machine learning that defines its own dataset format. This repository showcases how to integrate LeRobot datasets with Roboto's data management and processing platform through custom Actions.

## Actions

### Enrich LeRobot Dataset With Derived Data

An example Roboto Action that demonstrates how to:
- Load LeRobot datasets from Roboto storage
- Process and enrich dataset features with derived data
- Create new Roboto datasets with the enriched results

This action adds an `action_observation_difference` feature to LeRobot datasets, computing the element-wise difference between action and observation state vectors for each frame.

See the [action-specific README](./enrich-lerobot-dataset/README.md) for detailed setup and usage instructions.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) for building and running action images
- [Roboto CLI](https://github.com/roboto-ai/roboto-python-sdk/tree/main?tab=readme-ov-file#install-roboto) for deploying actions to the Roboto platform
- Python 3.10 or higher

## Resources

- [Roboto Platform Documentation](https://docs.roboto.ai/)
- [Roboto Actions Guide](https://docs.roboto.ai/user-guides/process-data-actions.html)
- [LeRobot Dataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
