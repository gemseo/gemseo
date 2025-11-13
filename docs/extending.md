<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Extend GEMSEO features

The simplest way is to create a subclass associated to the feature you want to extend, respectively:

- for optimizers, inherit from [BaseOptimizationLibrary][gemseo.algos.opt.base_optimization_library.BaseOptimizationLibrary], and put the Python file in the **`src/gemseo/algos/opt`** package,
- for DOEs, inherit from [BaseDOELibrary][gemseo.algos.doe.base_doe_library.BaseDOELibrary], and put the Python file in the **`src/gemseo/algos/doe`** package,
- for surrogate models, inherit from [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor], and put the Python file in the **`src/gemseo/mlearning/regression/algos`** package,
- for MDAs, inherit from [BaseMDA][gemseo.mda.base_mda.BaseMDA], and put the Python file in the **`src/gemseo/mda`** package,
- for MDO formulations, inherit from [BaseMDOFormulation][gemseo.formulations.base_mdo_formulation.BaseMDOFormulation], and put the Python file in the **`src/gemseo/formulations`** package,
- for disciplines, inherit from [Discipline][gemseo.core.discipline.discipline.Discipline], and put the Python file in the **`src/gemseo/disciplines`** package, which you created.
- for job schedulers and HPC submission, inherit from [JobSchedulerDisciplineWrapper][gemseo.disciplines.wrappers.job_schedulers.discipline_wrapper.JobSchedulerDisciplineWrapper], and put the Python file in the **`src/gemseo/disciplines/wrappers/job_schedulers`** package, which you created.

GEMSEO features can be extended with external Python modules. All kinds of additional features can be implemented: disciplines, algorithms, formulations, post-processors, surrogate models, \... There are 2 ways to extend GEMSEO with Python modules:

- by creating a pip installable package with a setuptools entry point, see [BaseFactory][gemseo.core.base_factory] for more details,
- by setting the environment variable `GEMSEO_PATH` with the path to the directory that contains the Python modules, multiple directories can be separated by `:`.
