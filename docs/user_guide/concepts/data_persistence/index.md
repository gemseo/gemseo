---
reading_time: true
description: "Data persistence in GEMSEO covers the cache, the database and the dataset, three complementary tools to record, save and reuse the data produced by a process."
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

# Data persistence { #concept-data-persistence }

Many GEMSEO processes generate data that are relevant beyond their immediate run:
disciplinary evaluations, optimization histories, samples of a design of
experiments, gradients, observables.
Recording this data so that it can be queried, post-processed or reused later
is called *data persistence*.

It is needed to:

- skip redundant computations
  when an expensive evaluation has already been performed at a given input value,
- restart a sequential process from the last successful iteration,
- analyse, visualize or compare results after the run,
- feed surrogate models and other machine learning algorithms,
- exchange data between tools.

## Different types of data persistence { #concept-different-types-of-data-persistence }

GEMSEO offers three complementary tools for data persistence:

- the [Cache][concept-cache] stores the evaluations of a
  [Discipline][gemseo.core.discipline.discipline.Discipline]
  or of any process discipline that composes other disciplines, such as
  a [DisciplineChain][gemseo.discipline.chain.chain.DisciplineChain]
  or an [MDAChain][gemseo.mda.chain.MDAChain]
  (inputs, outputs and Jacobian),
- the [Database][concept-database] stores the evaluations of the
  [ArrayFunction][gemseo.core.function.array_function.ArrayFunction] instances
  attached to an
  [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem]
  (objective, constraints, observables and their gradients),
- the [Dataset][concept-dataset] is a generic structure
  consolidating data into a multi-indexed table for post-processing,
  visualization and machine learning.

Caches and databases capture data *during* the execution of a process,
while datasets are typically built *from* them
to feed downstream analyses.
Any [Cache][gemseo.core.cache.base.BaseCache] or
[Database][gemseo.core.problem.database.Database]
can be exported to a [Dataset][gemseo.dataset.dataset.Dataset];
a dataset can also be created from a NumPy array, a CSV file or a pandas DataFrame.

## Cache, database and dataset at a glance { #concept-data-persistence-comparison }

| Aspect | [Cache][concept-cache] | [Database][concept-database] | [Dataset][concept-dataset] |
|---|---|---|---|
| Typical use | Skip redundant simulations, checkpoint a long sequential run | Optimization history, avoid double evaluating objective and constraints | Post-processing, visualization, surrogate model training |
| Scope | Any [Discipline][gemseo.core.discipline.discipline.Discipline], including process disciplines (chains, MDAs, ...) | An [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem] or [EvaluationProblem][gemseo.core.problem.evaluation.EvaluationProblem] | Standalone container |
| What is stored | Inputs, outputs and Jacobian of [execute()][gemseo.core.discipline.discipline.Discipline.execute] calls | Inputs, output values and gradients of the problem functions (objective, constraints, observables) | Any tabular data organised by group, variable and component |
| Input lookup | XXH64 hash of the flattened inputs, with optional tolerance | NumPy-array equality on the design vector | - |
| Persistence on disk | HDF5 ([HDF5Cache][gemseo.core.cache.hdf5.HDF5Cache]) | HDF5 ([Database.to_hdf()][gemseo.core.problem.database.Database.to_hdf] / [from_hdf()][gemseo.core.problem.database.Database.from_hdf]) | CSV, text, pandas [DataFrame][pandas.DataFrame] |
| Built from | Discipline executions | Solver / DOE iterations | A cache, a database, a NumPy array, a CSV file, a [DataFrame][pandas.DataFrame] |
| Convert to a [Dataset][concept-dataset] | [BaseCache.to_dataset()][gemseo.core.cache.base.BaseCache.to_dataset] | [Database.to_dataset()][gemseo.core.problem.database.Database.to_dataset] or [OptimizationProblem.to_dataset()][gemseo.optimization.problem.OptimizationProblem.to_dataset] | — |

## When to use which? { #concept-data-persistence-decision }

- *I want to skip re-running an expensive discipline at points where it
  has already been evaluated, or to checkpoint a long sequential
  process* → use a [Cache][concept-cache].
- *I want to record the iteration history of an optimization or a DOE,
  inspect it, post-process it or save it to disk* → use a
  [Database][concept-database]
  (one is already attached to any
  [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem]).
- *I want to plot, analyse or feed simulation or optimization data
  to a machine learning model* → build a
  [Dataset][concept-dataset]
  from the cache or the database.
- *I want to share or archive the data produced by a run* → persist to
  HDF5 via an [HDF5Cache][gemseo.core.cache.hdf5.HDF5Cache], a
  [Database.to_hdf()][gemseo.core.problem.database.Database.to_hdf]
  export, or a
  [Dataset][gemseo.dataset.dataset.Dataset]
  written through its underlying pandas
  [DataFrame][pandas.DataFrame].
