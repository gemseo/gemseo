<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<div align="center">

<img src="https://gitlab.com/gemseo/dev/gemseo/-/raw/develop/docs/assets/images/gemseo/logo-small.png" alt="GEMSEO logo" width="420"/>

Generic Engine for Multidisciplinary Scenarios, Exploration and Optimization

[![PyPI](https://img.shields.io/pypi/v/gemseo)](https://pypi.org/project/gemseo/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/gemseo)](https://anaconda.org/conda-forge/gemseo)
![Python](https://img.shields.io/pypi/pyversions/gemseo)
[![License: LGPL v3](https://img.shields.io/pypi/l/gemseo)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![Coverage](https://img.shields.io/codecov/c/gitlab/gemseo:dev/gemseo/develop)](https://app.codecov.io/gl/gemseo:dev/gemseo/branch/develop)
[![DOI](https://img.shields.io/badge/DOI-10.2514%2F6.2018--0657-blue)](https://arc.aiaa.org/doi/10.2514/6.2018-0657)

[Documentation](https://gemseo.org/doc/gemseo) ·
[Community](https://gemseo.discourse.group) ·
[Issues](https://gitlab.com/gemseo/dev/gemseo/-/issues) ·
[Contact](mailto:contact@gemseo.org)

</div>

---

GEMSEO<sup>&reg;</sup> is an open-source Python library to automate multidisciplinary processes,
starting with multidisciplinary design optimization (MDO). It provides a catalog of MDO
formulations that assemble disciplines, multidisciplinary analyses and optimization or DOE
algorithms into an executable process.

GEMSEO can be used standalone or embedded in a simulation platform. Disciplines can wrap
Python code, Matlab, Scilab, Excel spreadsheets or executables. It is built on NumPy, SciPy
and Matplotlib, and distributed under the GNU LGPL v3.0 license, which allows commercial use.

## Features

- Automatic creation and execution of MDO processes from a chosen MDO formulation
  (MDF, IDF, BiLevel, disciplinary optimization).
- Numerical tools for multidisciplinary problems: coupling (MDA), optimization, design of
  experiments, linear solvers, ordinary differential equations, surrogate models, machine
  learning, uncertainty quantification and visualization.
- Disciplines can wrap Python, Matlab, Scilab, Excel spreadsheets or executables.
- A stable high-level API.
- Extensible through plugins and entry points.

The complete list of features and algorithms is available in the
[documentation](https://gemseo.org/doc/gemseo).

## Installation

```bash
pip install "gemseo[all]"
```

With conda:

```bash
conda create -c conda-forge -n gemseo gemseo
```

With uv:

```bash
uv pip install "gemseo[all]"
```

Check the installation:

```bash
python -c "import gemseo"
```

## Quickstart

See the "GEMSEO in 10 minutes" tutorial in the
[documentation](https://gemseo.gitlab.io/dev/gemseo/develop/generated/examples/tutorials/basic/plot_gemseo_in_10_minutes/) for a step-by-step example.

## Documentation and examples

The [full documentation](https://gemseo.org/doc/gemseo) includes installation, user guide, API reference and a gallery of
runnable examples.

## Plugins

GEMSEO can be extended with multiple [plugins](https://gemseo.org/plugins/available/) that add disciplines,
algorithms and formulations,
such as `gemseo-matlab`,
`gemseo-jax`,
`gemseo-umdo`, ...

## Citation

If you use GEMSEO in your research, please cite:

```bibtex
@inbook{doi:10.2514/6.2018-0657,
author = {Francois Gallard and Charlie Vanaret and Damien Guenot and Vincent Gachelin and Rémi Lafage and Benoit Pauwels and Pierre-Jean Barjhoux and Anne Gazaix},
title = {GEMS: A Python Library for Automation of Multidisciplinary Design Optimization Process Generation},
booktitle = {2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference},
chapter = {},
pages = {},
doi = {10.2514/6.2018-0657},
URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2018-0657},
eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2018-0657}
}
```

## Contributing and community

- [Contributing guide](https://gemseo.org/contribute/contributing/)
- [Community forum](https://gemseo.discourse.group)
- [Bugs and questions](https://gitlab.com/gemseo/dev/gemseo/-/issues)
- [About and history](https://gemseo.org/about/history/)

## License

GEMSEO is distributed under the GNU LGPL v3.0 license (see `LICENSE.txt`);
the GNU GPL v3.0 reference is in the `LICENSES/` folder.
The examples use the BSD 0-Clause license.
The documentation uses the CC BY-SA 4.0 license.
The list of third-party dependencies and their licenses
is on the credits page of the documentation.
