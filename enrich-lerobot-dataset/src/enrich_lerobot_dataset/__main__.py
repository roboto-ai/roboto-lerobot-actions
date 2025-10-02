"""
Used as the entrypoint script when run in a Docker container.
Requires a number of environment variables to be set,
which are done automatically by Roboto when run on hosted compute.
"""

import pathlib
import tempfile

import roboto

from .entrypoint import Args, main

action_env = roboto.env.RobotoEnv.default()
action_runtime = roboto.ActionRuntime.from_env()


def get_optional_parameter(name: str) -> str | None:
    parameter_env_name = roboto.env.RobotoEnvKey.for_parameter(name)
    return action_env.get_env_var(parameter_env_name)


with tempfile.TemporaryDirectory() as tmpdir:
    args = Args(
        # Action Parameters are defined in `action.json`
        episode_limit=get_optional_parameter("EPISODE_LIMIT"),
        lerobot_source_dataset_path=action_runtime.input_dir,
        output_dir=pathlib.Path(tmpdir),
        roboto_source_dataset_id=action_runtime.dataset_id,
        roboto_org_id=action_runtime.org_id,
    )

    main(args)
