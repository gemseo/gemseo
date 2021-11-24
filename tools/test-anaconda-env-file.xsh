#! /usr/bin/env xonsh

# this script creates anaconda environments from environment files,
# installs gemseo and run the tests

trace on
$XONSH_TRACE_SUBPROC = True
$RAISE_SUBPROC_ERROR = True

for python_version in ("2",):
    # path to the environment
    env_path = f"{$ENVTMPDIR}/{python_version}"

    # create the env
    conda env create \
        -p @(env_path) \
        -f environment-py@(python_version).yml

    # install gemseo, pytest and run the tests
    for command in ("pip install .[all,test]", "pytest"):
        conda run \
            -p @(env_path) \
            --no-capture-output \
            @(command.split())
