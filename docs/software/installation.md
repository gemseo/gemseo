---
status: draft
description: ""
tags: [ ]
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

You may install the core or the full features set of GEMSEO. There are different ways to install GEMSEO, they are
described below.

## Requirements

We try to support the Python interpreter versions following the official
[Python release cycle](https://devguide.python.org/versions/#python-release-cycle).
We may not support newer version of Python if a dependency of GEMSEO does not support it.
The currently supported versions of Python are 3.10, 3.11, 3.12, 3.13 and 3.14.

To install GEMSEO, you should use a Python environment.
You can create environments with [uv](https://docs.astral.sh/uv),
or with the Python built-in [venv](https://docs.python.org/3/library/venv.html) module
or with [Anaconda](https://docs.anaconda.com/anaconda/install).

For using the full features set, if you are not using [Anaconda](https://docs.anaconda.com/anaconda/install), make sure
that [graphviz](https://graphviz.org/download) is installed (for rendering graphs).

## Dependencies

GEMSEO depends on external packages, some of them are optional. You may use more recent versions of these packages, but
we cannot guarantee the backward compatibility. However, we have a large set of tests with a high code coverage so that
you can fully check your configuration.

!!! info "See Also"

    Fully check your configuration with [Test with unit tests][test-with-unit-tests].

### Core features

The required dependencies provide the core features of GEMSEO, these are defined the `dependencies` entry
of [pyproject.toml](https://gitlab.com/gemseo/dev/gemseo/-/blob/master/pyproject.toml).

The minimal dependencies will allow to execute [MDO processes][mdo-formulations] but not all post processing tools will be available.

### Full features

Some packages are not required to execute basic scenarios, but provide additional features. The dependencies are
independent, and can be installed one by one to activate the dependent features of listed in the same table. Installing
all those dependencies will provide the full features set of GEMSEO. All these tools are open source with non-viral
licenses (see [Credits](../credits.md)), they are defined in the `all` entry of the `[project.optional-dependencies]`
section of [pyproject.toml](https://gitlab.com/gemseo/dev/gemseo/-/blob/master/pyproject.toml).

## Install from

=== "Pypi"

    Install the core features of the latest version with

    ``` console
    pip install gemseo
    ```

    or the full features with:

    ``` console
    pip install gemseo[all]
    ```

    See [pip](https://pip.pypa.io/en/stable/getting-started/) for more information.

=== "Anaconda"

    Install the full features in an anaconda environment named *gemseo* for Python 3.10 with

    ``` console
    conda create -c conda-forge -n gemseo python=3.10 gemseo
    ```

    You can change the Python version to 3.11, 3.12, 3.13 or 3.14.

=== "without internet access"

    If for some reasons you do not have access to internet from the target machine, such as behind a corporate firewall, you can use a [self-contained installer](https://mdo-ext.pf.irt-saintexupery.com/gemseo-installers).

## Test the installation

### Basic test

To check that the installation is successful, try to import the module:

``` console
python -c "import gemseo"
```

!!! warning
    If you obtain the error:

    ``` console
    “Traceback (most recent call last): File “<string>”, line 1, in <module> ImportError: No module named gemseo“
    ```

then the installation failed.

You can use the function [print_configuration()][gemseo.print_configuration] to print the successfully loaded modules
and the failed imports with the reason.

``` python
from gemseo import print_configuration

print_configuration()
```

This function is useful when only some of the GEMSEO features appear to be missing. Usually this is related to external
libraries that were not installed because the user did not request full features. See [Dependencies][dependencies] for
more information.

### Test with examples

The tutorials and how-tos contain many examples
to illustrate the main features of GEMSEO.
For each example, you can download a Python script or a Jupyter Notebook,
execute it and experiment to test the installation.

## Advanced

### Install the development version

Install the core features of the development version with

``` console
pip install gemseo@git+https://gitlab.com/gemseo/dev/gemseo.git@develop
```

or the full features with:

``` console
pip install gemseo[all]@git+https://gitlab.com/gemseo/dev/gemseo.git@develop
```

To develop in GEMSEO, see
instead [Developer information](https://gemseo.gitlab.io/dev/gemseo-org/develop/software/developing).

### Test with unit tests

Run the tests with:

``` console
pip install gemseo[all,test]
```

Look at the output of the above command to determine the installed version of GEMSEO. Get the tests corresponding to the
same version of GEMSEO from [gitlab](https://gitlab.com/gemseo/dev/gemseo). Then from the directory of this archive that
contains the `tests` directory, run

``` console
pytest
```

Look at the [contributing](https://gemseo.gitlab.io/dev/gemseo-org/develop/developing) section for more information on
testing.
