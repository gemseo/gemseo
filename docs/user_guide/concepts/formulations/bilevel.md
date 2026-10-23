---
reading_time: true
complexity: advanced
status: draft
description: ""
tags: ['user_guide']
search:
  boost: 2
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# The Bi-level formulation { #concept-the-bi-level-formulation }

Bi-level formulations are a family of MDO formulations
that involve multiple optimization sub-problems to
be solved to obtain the solution of the MDO problem.

In many of them, and in particular in formulations derived from BLISS,
the optimization sub-problems are separated according to the design variables.
The design variables shared by multiple disciplines
are put in a so-called system level optimization sub-problem.
In so-called disciplinary optimization sub-problems,
only the design variables that have a direct impact on one discipline are used.
Then,
the coupling variables may be solved by a [MDA][concept-solving-multi-disciplinary-analysis],
as in formulations derived from MDF (BLISS, ASO or CSSO),
or by using consistency constraints (eventually handled through a penalty approach)
like in IDF-like formulations (CO or ATC).

The next figure shows
the decomposition of the Bi-level MDO formulation implemented in GEMSEO with two MDAs,
the parallel sub-optimizations
and a main optimization (system level) on the shared variables.
It is an MDF-based approach,
derived from the BLISS 98 formulation and variants from ONERA[@Blondeau2012].
This formulation was invented in the MDA-MDO project at
IRT Saint Exupéry[@gazaix2017towards][@Gazaix2019] and also used in the
R-EVOL project[@gazaix2024industrialization].

![A process based on a Bi-level formulation.](../../../assets/images/mdo_formulations/bilevel_process.png)

This block decomposition is motivated by several purposes.
First, this separation aligns
with the industrial needs of work repartition between domains,
which matches the decomposition in terms of disciplines.
It allows for greater flexibility
in the use of specific approaches (algorithms) for solving
disciplinary optimizations, also dealing with less design variables at the same time.
Secondly,
as the full coupled derivatives may not be available,
the use of a gradient-based
approach with all variables in the loop may not be affordable.

In the current Bi-level formulation, the objective function is minimized block by block,
in parallel,
with each block $i$ minimizing its own variables $x_i$
and handling its own constraints $g_i$.
Sometimes, if it is not straightforward to optimize the objective function $f$
in the sub-problem $i$, another function $f_i$ can be considered as long as its decay
is consistent with the decay (monotonic decrease) of the overall objective function $f$.
The decomposition is such that the sub-problems constraints $g_i$ are assumed
to depend on other block variables $x_{\neq i}$ only through the couplings.
These couplings are solved by two MDAs: one before the sub-optimizations in order
to compute equilibrium values for each block,
and the second one after the sub-optimizations in order to recompute the equilibrium
for system level functions.
The sub-optimization blocks do not exchange any information
when they are solved in parallel,
which means that the synchronization is ensured by the two MDAs
and the system iterations
which warm start each block with the previous optimal values of local variables.
If the effect of one block variables $x_i$ on another block $j$ is too significant,
it means that the optimal solution $x^*$ is sensitive to the initial guess $x$,
and therefore that for same values of shared variables $z$,
different solutions $x^*$ can be obtained.
As a consequence,
the synchronization mechanism may not be sufficient
to solve accurately the lower problem and the system level algorithm may not converge
to the right solution.
In such a situation,
an enhancement is proposed with the
[bi-level BCD formulation][concept-the-bi-level-block-coordinate-descent-formulation]
which extends the range of problems that can be solved with Bi-level approaches.

## Post-processing the sub-scenario histories { #concept-post-processing-the-sub-scenario-histories }

When `BiLevel_Settings.keep_opt_history`
and/or `BiLevel_Settings.save_opt_history` are enabled,
the optimization history of each sub-scenario is retained
after every execution of the sub-scenario.
`BiLevelScenarioResult.get_sub_scenario_history_dataset`
combines that retained history,
across all these executions,
into a single [Dataset][gemseo.dataset.dataset.Dataset]
— one row per sub-scenario iteration,
tagged with the execution number,
the sub-scenario iteration number
and the upper-level values passed to the sub-scenario for that execution:

```python
scenario.execute()
scenario_result = scenario.get_result()
history_dataset = scenario_result.get_sub_scenario_history_dataset(0)
```

As this dataset stacks several independent optimization histories,
it is a plain [Dataset][gemseo.dataset.dataset.Dataset]
carrying no optimization metadata;
it is therefore not meant to be passed to [execute_post][gemseo.execute_post].

!!! warning
    The sub-scenario histories are not collected
    when the sub-scenarios are executed in separate processes,
    for example with `BiLevel_Settings(parallel_scenarios=True, multithread_scenarios=False)`,
    or with a system-level DOE that uses several processes:
    `get_sub_scenario_history_dataset` then raises a `ValueError`,
    even if the files of `save_opt_history` are written to the disk.

!!! warning
    With `save_opt_history` alone,
    the histories are read back from the HDF5 files.
    An overwrite of these files,
    e.g. by another scenario exporting to the same path,
    is detected through their modification time.
    This may miss a rewrite made within the timestamp resolution of the filesystem,
    e.g. 1-2 s on FAT, HFS+, ext3 or some network mounts,
    and the history of another run is then returned without error.
    When a file was deleted or moved,
    `get_sub_optimization_result` logs a warning and returns `None`,
    and `get_sub_scenario_history_dataset` raises a `ValueError`,
    so keep the files in place until the post-processing is done.
    The files are written in the current working directory
    and named after the scenario, the sub-scenario index and `BiLevel_Settings.naming`.
    Execute each scenario from its own directory,
    use `naming=NameGenerator.Naming.UUID`,
    or enable `keep_opt_history`.

The columns are grouped as follows:

| Group | Variables | Meaning |
|---|---|---|
| `designs` | the sub-scenario design variables | one column per component |
| `objectives` | the sub-scenario objective(s) | from `OptimizationProblem.to_dataset` |
| `equality_constraints`, `inequality_constraints` | the sub-scenario constraints, if any | from `OptimizationProblem.to_dataset` |
| `observables` | the sub-scenario observables, if any | from `OptimizationProblem.to_dataset` |
| `executions` (`BiLevelScenarioResult.executions_group`) | `execution`, `sub_iteration` | 1-based execution and iteration numbers |
| `upper_level_designs` (`BiLevelScenarioResult.upper_level_designs_group`) | the upper-level variables | the values passed to the sub-scenario for that execution |

See
[Post-process the sub-scenario histories of a bi-level scenario][post-process-the-sub-scenario-histories-of-a-bi-level-scenario]
for a worked example
that builds this dataset, extracts subsets of it
and plots the sub-scenario convergence across executions.

## Going further { #concept-going-further }

!!! tip "How-tos"
    - [MDO formulation][mdo-formulation]
    - [Post-process the sub-scenario histories of a bi-level scenario][post-process-the-sub-scenario-histories-of-a-bi-level-scenario]
