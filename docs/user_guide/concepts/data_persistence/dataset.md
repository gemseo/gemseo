---
description: "A dataset is a generic multi-indexed table grouping variables by category, which consolidates data from caches, databases or files for post-processing, visualization and machine learning."
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

# Dataset { #concept-dataset }

A [Dataset][gemseo.datasets.dataset.Dataset]
is a generic data structure that organizes
heterogeneous data into a single table.
It is backed by a pandas
[DataFrame][pandas.DataFrame]
whose columns follow a three-level hierarchy
`(GROUP, VARIABLE, COMPONENT)`:

- a *variable* is a named quantity with one or several *components*
  (for instance the 2-component variable `x` with components `0` and `1`,
  or the 4-component variable `y` with components `"a"`, `"b"`, `"c"` and `"d"`);
- a variable belongs to a *group*
  such as `"inputs"`, `"outputs"`, `"designs"`, `"objectives"`...
  Two variables may share a name across different groups,
  so the unique identifier of a variable
  is the tuple `(group_name, variable_name)`;
- each row of the dataset is an *entry*
  (a sample, an iteration or an observation depending on the context).

Free-form metadata that is not specific to an entry
can be stored in the
[misc][gemseo.datasets.dataset.Dataset.misc] dictionary,
for instance `dataset.misc["year"] = 2023`.

A [Dataset][gemseo.datasets.dataset.Dataset]
is the central data exchange structure of GEMSEO:
it is consumed by machine learning models,
by the dataset-based
[BasePost][gemseo.post.base_post.BasePost] subclasses
and by the [dataset visualization gallery][example-dataset-visualization].

## How it works { #concept-dataset-how-it-works }

A [Dataset][gemseo.datasets.dataset.Dataset]
is typically built incrementally
with
[add_variable()][gemseo.datasets.dataset.Dataset.add_variable]
or
[add_group()][gemseo.datasets.dataset.Dataset.add_group],
or loaded from an external source with one of the dedicated factory methods:

- [from_array()][gemseo.datasets.dataset.Dataset.from_array]
  for a NumPy array,
- [from_csv()][gemseo.datasets.dataset.Dataset.from_csv]
  for a CSV file,
- [from_txt()][gemseo.datasets.dataset.Dataset.from_txt]
  for a simple text file,
- [from_dataframe()][gemseo.datasets.dataset.Dataset.from_dataframe]
  for a pandas
  [DataFrame][pandas.DataFrame]
  with tuple or MultiIndex columns.

Slices and projections are obtained with
[get_view()][gemseo.datasets.dataset.Dataset.get_view],
which selects groups, variables, components or entries.
Conversions to other structures are available,
such as
[to_dict_of_arrays()][gemseo.datasets.dataset.Dataset.to_dict_of_arrays]
which returns nested or flat dictionaries of NumPy arrays.
Introspection is supported through properties
[group_names][gemseo.datasets.dataset.Dataset.group_names],
[variable_names][gemseo.datasets.dataset.Dataset.variable_names]
and
[variable_identifiers][gemseo.datasets.dataset.Dataset.variable_identifiers],
and via the
[summary][gemseo.datasets.dataset.Dataset.summary] string.

!!! warning
    A [Dataset][gemseo.datasets.dataset.Dataset]
    behaves like any multi-index
    [DataFrame][pandas.DataFrame],
    but instantiating one directly with the constructor
    `dataset = Dataset(data, ...)`
    can lead to inconsistencies
    (multi-index levels, index values, dtypes...).
    Building it with the dedicated methods is recommended,
    for instance
    `dataset = Dataset(); dataset.add_variable("x", data)`.

## Specialized datasets { #concept-specialized-datasets }

GEMSEO ships with two specialized
[Dataset][gemseo.datasets.dataset.Dataset] subclasses
that fix the group names
to the conventions most commonly used in MDO:

- [IODataset][gemseo.datasets.io_dataset.IODataset]
  separates `"inputs"` from `"outputs"`
  and exposes the convenience builders
  [add_input_variable()][gemseo.datasets.io_dataset.IODataset.add_input_variable]
  and
  [add_output_variable()][gemseo.datasets.io_dataset.IODataset.add_output_variable].
  It is the standard structure for surrogate model training.
- [OptimizationDataset][gemseo.datasets.optimization_dataset.OptimizationDataset]
  uses the groups `"designs"`, `"objectives"`,
  `"inequality_constraints"`, `"equality_constraints"` and `"observables"`,
  and adds iteration-oriented accessors
  ([n_iterations][gemseo.datasets.optimization_dataset.OptimizationDataset.n_iterations],
  [iterations][gemseo.datasets.optimization_dataset.OptimizationDataset.iterations],
  [design_variable_names][gemseo.datasets.optimization_dataset.OptimizationDataset.design_variable_names]).
  It is typically produced by
  [OptimizationProblem.to_dataset()][gemseo.algos.optimization_problem.OptimizationProblem.to_dataset].

Any [Dataset][gemseo.datasets.dataset.Dataset] subclass
is automatically discovered by the
[DatasetFactory][gemseo.datasets.factory.DatasetFactory],
so user-defined specializations can be plugged in
without changing the calling code.

## Creating a dataset { #concept-dataset-creation }

A [Dataset][gemseo.datasets.dataset.Dataset] can be built
from any of the standard data persistence containers of GEMSEO
or from external files:

- from a NumPy array, with
  [Dataset.from_array()][gemseo.datasets.dataset.Dataset.from_array];
- from a discipline [Cache][concept-cache], with
  [BaseCache.to_dataset()][gemseo.caches.base.BaseCache.to_dataset];
- from a [Database][concept-database], with
  [Database.to_dataset()][gemseo.algos.database.Database.to_dataset]
  or the higher-level
  [OptimizationProblem.to_dataset()][gemseo.algos.optimization_problem.OptimizationProblem.to_dataset];
- from CSV or text files, with
  [Dataset.from_csv()][gemseo.datasets.dataset.Dataset.from_csv]
  and
  [Dataset.from_txt()][gemseo.datasets.dataset.Dataset.from_txt].

!!! how-to
    - [How to create a dataset from a NumPy array][how-to-create-a-dataset-from-a-numpy-array]
    - [Convert a cache to a dataset][convert-a-cache-to-a-dataset]
    - [Convert a database to a dataset][convert-a-database-to-a-dataset]

## Visualization { #concept-dataset-visualization }

A [Dataset][gemseo.datasets.dataset.Dataset]
can be visualized through the
[BaseDatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot] hierarchy,
which offers a wide variety of plot types,
including
[Lines][gemseo.post.dataset.lines.Lines],
[Scatter][gemseo.post.dataset.scatter.Scatter],
[PairPlot][gemseo.post.dataset.pair_plot.PairPlot],
[ParallelCoordinates][gemseo.post.dataset.parallel_coordinates.ParallelCoordinates],
[AndrewsCurves][gemseo.post.dataset.andrews_curves.AndrewsCurves],
[RadarChart][gemseo.post.dataset.radar_chart.RadarChart],
[Boxplot][gemseo.post.dataset.boxplot.Boxplot],
[BarPlot][gemseo.post.dataset.bars.BarPlot],
[YvsX][gemseo.post.dataset.yvsx.YvsX]
and
[ZvsXY][gemseo.post.dataset.zvsxy.ZvsXY].
Plots can be combined into a single figure,
customized through the underlying matplotlib API
or rendered as interactive HTML through plotly.
See the [Dataset visualization gallery][example-dataset-visualization]
for the full collection of examples.

## Going further { #concept-going-further }

The following concepts are related to the dataset:

- [Cache][concept-cache] and [Database][concept-database]:
  the two main producers of datasets.
- [Machine learning](../machine_learning.md):
  datasets are the standard input
  of GEMSEO's machine learning models.
