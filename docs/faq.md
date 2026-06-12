---
status: draft
description: ""
tags: []
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

# Frequently Asked Questions

## Upgrading GEMSEO { #upgrading-gemseo-faq }

As GEMSEO code evolves, some calling signatures and behavior may change. These changes may break the codes that use GEMSEO and require modifications of them. See [Upgrading GEMSEO](software/upgrading.md) for more information.

## Create a simple DOE on a single discipline

Use the [DisciplinaryOpt][gemseo.formulations.disciplinary_opt.DisciplinaryOpt] formulation and a [MDOScenario][gemseo.scenarios.mdo.MDOScenario] scenario. Even for simple DOEs, GEMSEO formulates an optimization problem, so requires a [MDO formulation][concept-mdo-formulations]. The [DisciplinaryOpt][gemseo.formulations.disciplinary_opt.DisciplinaryOpt] formulation executes the [Discipline][gemseo.core.discipline.discipline.Discipline] alone, or the list of [Discipline][gemseo.core.discipline.discipline.Discipline] in the order passed by the user. This means that you must specify an objective function which will be set for the visualizations. The minimum value through the DOE will be considered as the best one.

!!! info "See Also"
    For more details, we invite you to read [our tutorial][tutorial-execute-your-first-design-of-experiment-doe].

## Create a simple optimization on a single discipline

Use the [DisciplinaryOpt][gemseo.formulations.disciplinary_opt.DisciplinaryOpt] formulation and an [MDOScenario][gemseo.scenarios.mdo.MDOScenario]. The [DisciplinaryOpt][gemseo.formulations.disciplinary_opt.DisciplinaryOpt] formulation executes the [Discipline][gemseo.core.discipline.discipline.Discipline] alone, or the list of [Discipline][gemseo.core.discipline.discipline.Discipline] in the order passed by the user.

## Available options for algorithms

See the available [DOEs][available-doe-algorithms],
[linear solvers][available-linear-solvers],
[MDO formulations][available-linear-solvers],
[MDAs][available-mda-algorithms],
[optimizers][available-optimization-algorithms], [post-processors][available-post-processing-algorithms]
and [machine learners][algorithms-of-machine-learning] (accessible from [this page][overview] of the documentation).

## How to extend GEMSEO features?

See [Extending Gemseo][extensibility].

## What are JSON schemas?

JSON schemas describe the format (i.e. structure) of JSON files, in a similar way as XML schemas define the format of XML files. JSON schemas come along with validators, that check that a JSON data structure is valid against a JSON schema, this is used in GEMSEO' Grammars.

!!! info "See Also"
    We invite you to read our documentation: ["Input and output description: grammars"][concept-grammars].

!!! info "See Also"
    The details about the [JSON schema specification](https://json-schema.org/docs).

## Store persistent data produced by disciplines

Use HDF5 caches to persist the discipline output on the disk.

!!! info "See Also"
    We invite you to read our documentation: [Cache](user_guide/concepts/data_persistence/cache.md).

## Error when using a HDF5 cache

In GEMSEO 3.2.0, the storage of the data hashes in the HDF5 cache has been fixed and the previous cache files are no longer valid. If you get an error like `The file cache.h5 cannot be used because it has no file format version: see HDF5Cache.update_file_format for converting it.`, please use [HDF5Cache.update_file_format()][gemseo.caches.hdf5.HDF5Cache.update_file_format] to update the format of the file and fix the data hashes.

## GEMSEO fails with openturns

Openturns implicitly requires the library *libnsl* that may not be installed by default on recent linux OSes. Under *CentOS* for instance, install it with:

``` console
sudo yum install libnsl
```

## Parallel execution limitations on Windows

When running parallel execution tasks on Windows, the [HDF5Cache][gemseo.caches.hdf5.HDF5Cache] does not work properly. This is due to the way subprocesses are forked in this architecture. The method [MDOScenario.set_backup_settings()][gemseo.scenarios.mdo.MDOScenario.set_backup_settings] is recommended as an alternative.

The execution of any script using parallel execution on Windows including, but not limited to, [MDOScenario][gemseo.scenarios.mdo.MDOScenario] with `n_processes > 1`, [HDF5Cache][gemseo.caches.hdf5.HDF5Cache], [MemoryFullCache][gemseo.caches.memory_full.MemoryFullCache], [CallableParallelExecution][gemseo.core.parallel_execution.callable_parallel_execution.CallableParallelExecution], [DiscParallelExecution][gemseo.core.parallel_execution.discipline_execution.DiscParallelExecution], must be protected by an `if __name__ == '__main__':` statement.

## Handling paths for different OSes

Some disciplines wrap other disciplines in order to execute them remotely. Those disciplines may use paths stored as [Path][pathlib.Path], which are handled differently on Windows and on POSIX platforms (Linux and MacOS). Despite the fact that GEMSEO takes care of converting those types of paths, it cannot convert absolute paths. For instance, in the path `C:\\some\path`, the `C:` part has no meaning on POSIX platforms. In that case, to prevent GEMSEO from terminating with an error, these types of paths should be defined as relative paths. For instance, the paths `some\path` or `some/path` are relative paths, which are relative to the current working directory.

## How to list the different features?

GEMSEO offers different `get_available_*` methods to list the different features.
They can be directly imported from GEMSEO.
For instance, `from gemseo import get_available_post_processings`
to list all the post-processings.
