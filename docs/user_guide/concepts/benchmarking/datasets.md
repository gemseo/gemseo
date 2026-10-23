---
reading_time: true
complexity: beginner
description: "Academic datasets bundled with GEMSEO for illustrating surrogate, machine learning and data analysis capabilities."
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

# Datasets { #concept-benchmarking-datasets }

GEMSEO provides a set of academic datasets
via the [gemseo.problem.dataset][gemseo.problem.dataset] package.
These datasets are ready-to-use [Dataset][gemseo.dataset.dataset.Dataset] instances
intended to illustrate and benchmark surrogate modelling, machine learning, and data
analysis capabilities.

## Burgers { #concept-burgers-dataset }

The Burgers dataset contains solutions to the viscous Burgers' equation

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2}$$

with periodic boundary conditions on $[0, 2\pi]$.
It is produced by a full-factorial design of experiments whose samples are time steps
and whose features are spatial grid points.
The default dataset has 30 time steps and 501 spatial points,
with fluid viscosity $\nu = 0.1$.

??? abstract "API"

    - [create_burgers_dataset][gemseo.problem.dataset.burgers.create_burgers_dataset]:
      returns an [IODataset][gemseo.dataset.io_dataset.IODataset]
      with inputs (time) and outputs (spatial field $u$).

## Iris { #concept-iris-dataset }

The Iris dataset is a classic benchmark for clustering and classification algorithms.
It contains 150 observations of iris plants split equally among three species:
*Iris setosa*, *Iris versicolour* and *Iris virginica*.
Each observation has four features: sepal length, sepal width, petal length and petal width.

??? abstract "API"

    - [create_iris_dataset][gemseo.problem.dataset.iris.create_iris_dataset]:
      returns a [Dataset][gemseo.dataset.dataset.Dataset].
      Set `as_io=True` to get an [IODataset][gemseo.dataset.io_dataset.IODataset]
      with features as inputs and species as output.

## Rosenbrock { #concept-rosenbrock-dataset }

The Rosenbrock dataset contains 100 evaluations of the Rosenbrock function

$$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$$

sampled on a regular grid.
The global minimum is at $(x, y) = (1, 1)$ with $f = 0$.
This dataset is useful for illustrating regression and surrogate modelling
on a function with a curved, narrow valley.

??? abstract "API"

    - [create_rosenbrock_dataset][gemseo.problem.dataset.rosenbrock.create_rosenbrock_dataset]:
      returns a [Dataset][gemseo.dataset.dataset.Dataset].
      Set `categorize=True` (default) to split into design and function groups.
