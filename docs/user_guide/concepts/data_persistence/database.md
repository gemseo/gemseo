---
reading_time: true
description: "A database stores the evaluations of the functions attached to an evaluation problem, both as an iteration history and as a lookup to avoid redundant computations."
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

# Database { #concept-database }

A [Database][gemseo.algos.database.Database]
stores the evaluations of functions:
inputs, output values and gradients.
It is attached
to an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
(or any [EvaluationProblem][gemseo.algos.evaluation_problem.EvaluationProblem],
see the [Evaluation problem page][concept-evaluation-problem])
to record the values of its objective, constraints and observables
along with their gradients,
either as an optimization history
or as a collection of samples in the case of a design of experiments (DOE).

A [Database][gemseo.algos.database.Database] plays two roles:

- it acts as an iteration history
  that can be inspected, post-processed
  or visualized — for instance with an
  [optimization history view][full-optimization-history-overview] —
  after the run;
- it avoids re-evaluating expensive functions
  at design points that have already been visited,
  which is especially useful
  when several functions share the same evaluation point.

!!! info "Database vs cache"
    A [Database][gemseo.algos.database.Database]
    is similar in spirit to a [Cache][concept-cache]
    but operates at a different scope:

    - a [cache][concept-cache] is *discipline-scoped*
      (the discipline can be a user discipline or a process discipline
      such as a chain or an MDA)
      and stores input/output/Jacobian values produced
      by [execute()][gemseo.core.discipline.discipline.Discipline.execute],
      with strict XXH64 hashing of flattened inputs;
    - a [Database][gemseo.algos.database.Database] is *problem-scoped*
      and stores the values of all functions
      attached to an
      [EvaluationProblem][gemseo.algos.evaluation_problem.EvaluationProblem],
      keyed by design vectors compared as NumPy arrays.

    Typically, the same [cache][concept-cache] attached to a discipline may be used
    through several types of analysis (DoE, different optimization scenarios, etc.),
    which makes it possible to store and aggregate all the executions of the same
    discipline.
    The [Database][gemseo.algos.database.Database] is solely related to
    a given scenario which means that one database will be built for each
    type of analysis.
    In case of large applications where the computational cost of the
    disciplines is important,
    it is advised to set a [HDF5Cache][gemseo.caches.hdf5.HDF5Cache].
    In such a situation,
    it will always be possible to rebuild the database
    from a new process execution,
    using the discipline caches.

!!! how-to
    - [Database examples][database-examples]

## Listeners { #concept-database-listeners }

User-defined callbacks, also called *listeners*,
can be triggered when storing a new entry in the database.
Typical use cases include live progress logging,
external monitoring
and custom convergence criteria.

!!! how-to
    - [Observe evaluations using listeners][observe-evaluations-using-listeners]

## Saving and loading { #concept-database-saving-loading }

A database can be persisted in HDF5 format.
This format is portable and well suited to large iteration histories
that must be revisited later for cold post-processing.

!!! how-to
    - [History backup][history-backup]

## Going further { #concept-going-further }

A [Database][gemseo.algos.database.Database]
can be converted into a
[Dataset][gemseo.datasets.dataset.Dataset]
for post-processing,
typically an
[OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
that splits the entries into design variables, objectives, constraints
and observables.
Two equivalent entry points are available:

- [Database.to_dataset()][gemseo.algos.database.Database.to_dataset]
  on the database directly,
- [OptimizationProblem.to_dataset()][gemseo.algos.optimization_problem.OptimizationProblem.to_dataset]
  on the surrounding problem.

!!! how-to
    - [Convert a database to a dataset][convert-a-database-to-a-dataset]

Finally, the following concepts are related to the database:

- [Optimization problem][concept-optimization-problem]:
  the typical owner of a database.
- [Evaluation problem][concept-evaluation-problem]:
  the more general problem class that also exposes a database.
- [Dataset][concept-dataset]:
  the structure that consolidates a database
  for post-processing, visualization and machine learning.
