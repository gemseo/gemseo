
..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _extending-gemseo:

Extend |g| features
-------------------

The simplest way is to create a subclass
associated to the feature you want to extend,
respectively:

- for optimizers,
  inherit from :class:`.BaseOptimizationLibrary`,
  and put the Python file in the :file:`src/gemseo/algos/opt` package,
- for DOEs,
  inherit from :class:`.BaseDOELibrary`,
  and put the Python file in the :file:`src/gemseo/algos/doe` package,
- for surrogate models,
  inherit from :class:`.BaseRegressor`,
  and put the Python file in the :file:`src/gemseo/mlearning/regression/algos` package,
- for MDAs, inherit from :class:`.BaseMDA`,
  and put the Python file in the :file:`src/gemseo/mda` package,
- for MDO formulations,
  inherit from :class:`.BaseMDOFormulation`,
  and put the Python file in the :file:`src/gemseo/formulations` package,
- for disciplines,
  inherit from :class:`.Discipline`,
  and put the Python file in the :file:`src/gemseo/disciplines` package,
  which you created.
- for job schedulers and HPC submission,
  inherit from :class:`.JobSchedulerDisciplineWrapper`,
  and put the Python file in the :file:`src/gemseo/disciplines/wrappers/job_schedulers` package,
  which you created.

|g| features can be extended with external Python modules.
All kinds of additional features can be implemented:
disciplines, algorithms, formulations, post-processors, surrogate models, ...
There are 2 ways to extend |g| with Python modules:

- by creating a pip installable package with a setuptools entry point,
  see :class:`.Factory` for more details,
- by setting the environment variable :envvar:`GEMSEO_PATH`
  with the path to the directory
  that contains the Python modules,
  multiple directories can be separated by :envvar:`:`.
