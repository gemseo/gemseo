<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Frequently Asked Questions

## Upgrading GEMSEO

As GEMSEO code evolves, some calling signatures and behavior may change. These changes may break the codes that use GEMSEO and require modifications of them. See [Upgrading GEMSEO](software/upgrading.md) for more information.

## Create a simple DOE on a single discipline

Use the [DisciplinaryOpt][gemseo.formulations.disciplinary_opt.DisciplinaryOpt] formulation and a [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario] scenario. Even for simple DOEs, GEMSEO formulates an optimization problem, so requires a [MDO formulation](mdo/mdo_formulations.md). The [DisciplinaryOpt][gemseo.formulations.disciplinary_opt.DisciplinaryOpt] formulation executes the [Discipline][gemseo.core.discipline.discipline.Discipline] alone, or the list of [Discipline][gemseo.core.discipline.discipline.Discipline] in the order passed by the user. This means that you must specify an objective function. The DOE won't try to minimize it but it will be set as an objective in the visualizations.

!!! info "See Also"
    For more details, we invite you to read [our example][mdf-based-doe-on-the-sobieski-ssbj-test-case].

## Create a simple optimization on a single discipline

Use the [DisciplinaryOpt][gemseo.formulations.disciplinary_opt.DisciplinaryOpt] formulation and an [MDOScenario][gemseo.scenarios.mdo_scenario.MDOScenario]. The [DisciplinaryOpt][gemseo.formulations.disciplinary_opt.DisciplinaryOpt] formulation executes the [Discipline][gemseo.core.discipline.discipline.Discipline] alone, or the list of [Discipline][gemseo.core.discipline.discipline.Discipline] in the order passed by the user.

## Available options for algorithms

See the available [DOEs][available-doe-algorithms],
[linear solvers][available-linear-solvers],
[MDO formulations][available-linear-solvers],
[MDAs][available-mda-algorithms],
[optimizers][available-optimization-algorithms], [post-processors][available-post-processing-algorithms]
and [machine learners][algorithms-of-machine-learning] (accessible from [this page][overview] of the documentation).

## Coupling a simulation software to GEMSEO

See [Interfacing simulation software](interface/software_connection.md).

!!! info "See Also"
    We invite you to discover all the steps in [this example][a-from-scratch-example-on-the-sellar-problem].

## How to extend GEMSEO features?

See [Extending Gemseo][extend-gemseo-features].

## What are JSON schemas?

JSON schemas describe the format (i.e. structure) of JSON files, in a similar way as XML schemas define the format of XML files. JSON schemas come along with validators, that check that a JSON data structure is valid against a JSON schema, this is used in GEMSEO' Grammars.

!!! info "See Also"
    We invite you to read our documentation: ["Input and output description: grammars"][input-and-output-description-grammars].

!!! info "See Also"
    The details about the [JSON schema specification](https://json-schema.org/docs).

## Store persistent data produced by disciplines

Use HDF5  caches to persist the discipline output on the disk.

!!! info "See Also"
    We invite you to read our documentation: [Cache](data_persistence/cache.md).

## Error when using a HDF5 cache

In GEMSEO 3.2.0, the storage of the data hashes in the HDF5 cache has been fixed and the previous cache files are no longer valid. If you get an error like `The file cache.h5 cannot be used because it has no file format version: see HDF5Cache.update_file_format for converting it.`, please use [HDF5Cache.update_file_format()][gemseo.caches.hdf5_cache.HDF5Cache.update_file_format] to update the format of the file and fix the data hashes.

## GEMSEO fails with openturns

Openturns implicitly requires the library *libnsl* that may not be installed by default on recent linux OSes. Under *CentOS* for instance, install it with:

``` console
sudo yum install libnsl
```

## Parallel execution limitations on Windows

When running parallel execution tasks on Windows, the [HDF5Cache][gemseo.caches.hdf5_cache.HDF5Cache] does not work properly. This is due to the way subprocesses are forked in this architecture. The method [DOEScenario.set_optimization_history_backup()][gemseo.scenarios.doe_scenario.DOEScenario.set_optimization_history_backup] is recommended as an alternative.

The execution of any script using parallel execution on Windows including, but not limited to, [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario] with `n_processes > 1`, [HDF5Cache][gemseo.caches.hdf5_cache.HDF5Cache], [MemoryFullCache][gemseo.caches.memory_full_cache.MemoryFullCache], [CallableParallelExecution][gemseo.core.parallel_execution.callable_parallel_execution.CallableParallelExecution], [DiscParallelExecution][gemseo.core.parallel_execution.disc_parallel_execution.DiscParallelExecution], must be protected by an `if __name__ == '__main__':` statement.

## Handling paths for different OSes

Some disciplines wrap other disciplines in order to execute them remotely. Those disciplines may use paths stored as [Path][pathlib.Path], which are handled differently on Windows and on POSIX platforms (Linux and MacOS). Despite the fact that GEMSEO takes care of converting those types of paths, it cannot convert absolute paths. For instance, in the path `C:\\some\path`, the `C:` part has no meaning on POSIX platforms. In that case, to prevent GEMSEO from terminating with an error, these types of paths should be defined as relative paths. For instance, the paths `some\path` or `some/path` are relative paths, which are relative to the current working directory.
