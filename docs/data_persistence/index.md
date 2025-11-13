
<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Contributors:
         :author: Matthias De Lozzo
-->

# Data persistence overview

Many processes generate data that can be relevant for post-processing
(surrogate modelling, data analysis, visualization, ...).
Recording these data is called *data persistence*
and allows access to it at any time.

GEMSEO offers different tools for data persistence:

- the [cache][gemseo.caches] stores the evaluations of a
  [Discipline][gemseo.core.discipline.discipline.Discipline],
- the [database][gemseo.algos.database] stores the evaluations of the
  [MDOFunction][gemseo.core.mdo_functions.mdo_function.MDOFunction] instances attached
  [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem],
- the [dataset][gemseo.datasets.dataset] is a generic structure facilitating the post-processing of the data. When solving [problems][gemseo.problems.dataset], a [dataset][gemseo.datasets.dataset] can be used for [visualization][example-dataset-visualization]. Some [examples][datasets] are given.
