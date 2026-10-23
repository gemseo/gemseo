---
description: "Install GEMSEO with pip, uv or conda, and check the installation."
tags: ['how_to']
search:
  boost: 1
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Installation

GEMSEO can be installed with its core features only or with its full feature set.
This page describes the different ways to install it and to check the installation.

## Requirements

GEMSEO supports the Python versions maintained according to the official
[Python release cycle](https://devguide.python.org/versions/#python-release-cycle),
as long as its dependencies support them.
It currently supports Python 3.11, 3.12, 3.13 and 3.14.

Install GEMSEO in a Python environment,
created for example with [uv](https://docs.astral.sh/uv),
the built-in [venv](https://docs.python.org/3/library/venv.html) module
or [Anaconda](https://docs.anaconda.com/anaconda/install).

For the full feature set without Anaconda,
also install [Graphviz](https://graphviz.org/download) to render graphs.
Its `dot` executable must be on the `PATH`;
under Windows, select the option adding Graphviz to the `PATH` when installing it.

## Dependencies

GEMSEO depends on external packages, some of which are optional.
More recent versions of these packages may work,
but their backward compatibility is not guaranteed;
the large test suite of GEMSEO lets you check your configuration.

!!! info "See Also"

    Fully check your configuration with [Test with unit tests][test-with-unit-tests].

### Core features

The required dependencies, listed in the `dependencies` entry
of [pyproject.toml](https://gitlab.com/gemseo/dev/gemseo/-/blob/master/pyproject.toml),
provide the core features of GEMSEO.
They let you execute [MDO processes][concept-mdo-formulations],
but some post-processing tools are not available.

### Full features

The optional dependencies, listed in the `all` entry of the `[project.optional-dependencies]` section
of [pyproject.toml](https://gitlab.com/gemseo/dev/gemseo/-/blob/master/pyproject.toml),
provide additional features.
They are independent of each other:
install one of them to activate the features depending on it,
or all of them to get the full feature set.
They are all open source with non-viral licenses (see [Credits](../credits.md)).

## Install from

=== "PyPI"

    === "pip"

        Install the core features of the latest version with

        ``` console
        pip install gemseo
        ```

        or the full feature set with

        ``` console
        pip install "gemseo[all]"
        ```

        See [pip](https://pip.pypa.io/en/stable/getting-started/) for more information.

    === "uv"

        Add the core features of the latest version to the dependencies of a uv project with

        ``` console
        uv add gemseo
        ```

        or the full feature set with

        ``` console
        uv add "gemseo[all]"
        ```

        To install GEMSEO in an existing environment instead,
        use `uv pip install gemseo` or `uv pip install "gemseo[all]"`.

        See [uv](https://docs.astral.sh/uv/) for more information.

=== "Anaconda"

    Create a conda environment named *gemseo* with Python 3.11 and the full feature set of GEMSEO with

    ``` console
    conda create -c conda-forge -n gemseo python=3.11 gemseo
    ```

    Replace 3.11 by any other [supported version][requirements] of Python.

## Test the installation

### Basic test

To check that the installation succeeded, import GEMSEO:

``` console
python -c "import gemseo"
```

or, in a uv project:

``` console
uv run python -c "import gemseo"
```

!!! warning

    If you obtain the error

    ``` console
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ModuleNotFoundError: No module named 'gemseo'
    ```

    then GEMSEO is not installed in the active Python environment.

To list the modules that were loaded and the imports that failed with their reasons,
use the function [print_configuration()][gemseo.print_configuration]:

``` python
from gemseo import print_configuration

print_configuration()
```

This is useful when only some features of GEMSEO are missing,
usually because the optional dependencies providing them are not installed;
see [Dependencies][dependencies].

### Test with examples

The [basic tutorials](../generated/examples/tutorials/basic/index.md),
the [advanced tutorials](../generated/examples/tutorials/advanced/index.md)
and the [how-to guides][disciplines-examples] illustrate the main features of GEMSEO.
Each example can be downloaded as a Python script or a Jupyter notebook,
which you can run and modify to test the installation.

## Advanced

### Install the development version

=== "pip"

    Install the core features of the development version with

    ``` console
    pip install "gemseo @ git+https://gitlab.com/gemseo/dev/gemseo.git@develop"
    ```

    or the full feature set with

    ``` console
    pip install "gemseo[all] @ git+https://gitlab.com/gemseo/dev/gemseo.git@develop"
    ```

=== "uv"

    Add the core features of the development version to the dependencies of a uv project with

    ``` console
    uv add "gemseo @ git+https://gitlab.com/gemseo/dev/gemseo.git@develop"
    ```

    or the full feature set with

    ``` console
    uv add "gemseo[all] @ git+https://gitlab.com/gemseo/dev/gemseo.git@develop"
    ```

    To install it in an existing environment instead, replace `uv add` by `uv pip install`.

To contribute to GEMSEO, see the [developer guide](https://gemseo.gitlab.io/dev/gemseo-org/contribute/developer_guide/) instead.

### Test with unit tests

Get the sources of the installed version of GEMSEO from [GitLab](https://gitlab.com/gemseo/dev/gemseo).
Then, from their root directory, which contains the `tests` directory, run the tests with

``` console
uv run --extra all pytest
```

(see [uv](https://docs.astral.sh/uv/) for more information) or, with pip 25.1 or newer, with

``` console
pip install --group test ".[all]"
pytest
```

See the [contributing guide](https://gemseo.gitlab.io/dev/gemseo-org/contribute/contributing/)
for more information on testing.
