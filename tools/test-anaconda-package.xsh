#! /usr/bin/env xonsh

# this script creates and tests the gemseo anaconda package
# we follow conda-forge channels best practices, see
# https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge

trace on
$XONSH_TRACE_SUBPROC = True
$RAISE_SUBPROC_ERROR = True

# clear the local repository
conda build purge

# build gemseo
# --override-channels: do not take into account the user's .condarc
conda build recipe \
    -c conda-forge \
    -c defaults \
    --override-channels \
    --python 3.8

for python_version in "3.6 3.7 3.8".split():
    # path to the environment
    env_path = f"{$ENVTMPDIR}/{python_version}"

    # create a test environment with the local gemseo and pytest
    # -c local or --use-local may not work: use -c ${CONDA_DEFAULT_ENV}/conda-bld
    # see https://github.com/conda/conda/issues/7024#issuecomment-431919976
    # --override-channels: do not take into account the user's .condarc
    conda create \
        -y \
        -p @(env_path) \
        -c conda-forge \
        -c defaults \
        -c $CONDA_DEFAULT_ENV/conda-bld \
        --override-channels \
        python=@(python_version) \
        pytest \
        gemseo

    # run the tests
    conda run \
        -p @(env_path) \
        --no-capture-output \
        pytest
