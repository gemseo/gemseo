<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Installation

You may install the core or the full features set of GEMSEO. See [Dependencies](dependencies.md) for more information. There are different ways to install GEMSEO, they are described below.

## Requirements

We try to support the Python interpreter versions following the official [Python release cycle](https://devguide.python.org/versions/#python-release-cycle). We may not support newer version of Python when a dependency of GEMSEO does not provide support for it. We may support an old version of Python from its end-of-life date until a major release of GEMSEO. The currently supported versions of Python are 3.9, 3.10, 3.11 and 3.12.

To install GEMSEO, you should use a Python environment. You can create environments with the Python built-in [venv](https://docs.python.org/3/library/venv.html) module or with [Anaconda](https://docs.anaconda.com/anaconda/install).

For using the full features set, if you are not using [Anaconda](https://docs.anaconda.com/anaconda/install), make sure that [graphviz](https://graphviz.org/download) is installed (for rendering graphs).

## Install from Pypi

Install the core features of the latest version with

``` console
pip install gemseo
```

or the full features with:

``` console
pip install gemseo[all]
```

See [pip](https://pip.pypa.io/en/stable/getting-started/) for more information.

## Install from Anaconda

Install the full features in an anaconda environment named *gemseo* for Python 3.9 with

``` console
conda create -c conda-forge -n gemseo python=3.9 gemseo
```

You can change the Python version to 3.10, 3.11 or 3.12.

## Install without internet access

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

You can use the function [print_configuration()][gemseo.print_configuration] to print the successfully loaded modules and the failed imports with the reason.

``` python
from gemseo import print_configuration

print_configuration()
```

This function is useful when only some of the GEMSEO features appear to be missing. Usually this is related to external libraries that were not installed because the user did not request full features. See [Dependencies](dependencies.md) for more information.

### Test with examples

The [gallery of examples][api] contains many examples to illustrate the main features of GEMSEO.
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

To develop in GEMSEO, see instead [Developer information][developer-information].

### Test with unit tests

Run the tests with:

``` console
pip install gemseo[all,test]
```

Look at the output of the above command to determine the installed version of GEMSEO. Get the tests corresponding to the same version of GEMSEO from [gitlab](https://gitlab.com/gemseo/dev/gemseo). Then from the directory of this archive that contains the `tests` directory, run

``` console
pytest
```

Look at the [contributing][developer-information] section for more information on testing.
