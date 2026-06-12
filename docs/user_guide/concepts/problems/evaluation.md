---
description: "An evaluation problem computes output values from input values by evaluating a collection of observable functions over a design space."
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

# Evaluation problem { #concept-evaluation-problem }

An [EvaluationProblem][gemseo.algos.evaluation_problem.EvaluationProblem] is the simplest form of computational problem.
It computes output values by evaluating a collection of [functions][concept-functions],
$f_1, \, \ldots, \, f_m$ called *observables*, such that $f_i: \mathbb{C}^d \to \mathbb{C}^{p_i}$,
on a given set of input values $\lbrace x_1,\, \ldots, \, x_n \rbrace$
defined on a [DesignSpace][gemseo.algos.design_space.DesignSpace].
It is formalized as,

$$
\left\{
\begin{matrix}
\; \text{Evaluate} \,\,\, & f_1(x), \, \ldots, \, f_m(x) \\[0.1cm]
\; \text{for} \,\,\, & x \in \lbrace x_1,\, \ldots, \, x_n \rbrace. \\
\end{matrix}
\right.
$$

The observables are [ArrayFunction][gemseo.core.functions.array_function.ArrayFunction] instances
added to the evaluation problem via [add_observable()][gemseo.algos.evaluation_problem.EvaluationProblem.add_observable].

!!! note

    An evaluation problem does not involve any optimization or equation solving.

    In particular, there is no notion of objective function or constraints.

Additionally, it can evaluate the Jacobian matrices $J_{f_1}\,(x), \, \ldots, \, J_{f_m}(x)$
for $x \in \lbrace x_1,\, \ldots, \, x_n \rbrace$, where

$$
J_{f_i}(x) \;=\;
\begin{bmatrix}
\dfrac{\partial f_{i,1}}{\partial x_1}(x) & \cdots & \dfrac{\partial f_{i,1}}{\partial x_d}(x) \\
\vdots & \ddots & \vdots \\
\dfrac{\partial f_{i,p_i}}{\partial x_1}(x) & \cdots & \dfrac{\partial f_{i,p_i}}{\partial x_d}(x)
\end{bmatrix}.
$$

The Jacobians can be either computed analytically or approximated using techniques such
as finite differences or complex step.

A [Database][gemseo.algos.database.Database] ([read more][concept-database]) can be passed at instantiation.
It will store the output values and the Jacobians to avoid re-evaluating functions at the same point.
User-defined callbacks can also be triggered when storing a new entry in the database.

Before being evaluated,
the functions are wrapped into a [ProblemFunction][gemseo.algos.problem_function.ProblemFunction] by
[preprocess_functions()][gemseo.algos.evaluation_problem.EvaluationProblem.preprocess_functions].
This wrapper takes care of input normalization, integer variable rounding, database caching,
and derivative approximation.
This preprocessing step is performed automatically before the algorithm execution begins.

The evaluation of the problem functions at a given design vector is done via
[evaluate_functions()][gemseo.algos.evaluation_problem.EvaluationProblem.evaluate_functions],
which returns the output values and optionally their Jacobian matrices.
The strategy to follow when an evaluation returns a `NaN` is determined by the boolean
attribute `stop_if_nan`.

A design of experiments (DOE) ([read more][concept-samplers-doe])
is typically used to process an [EvaluationProblem][gemseo.algos.evaluation_problem.EvaluationProblem],
that is, to evaluate its functions.

!!! note

    For post-processing purposes,
    the evaluations contained in the [Database][gemseo.algos.database.Database]
    can be exported to an HDF file or a [Dataset][gemseo.datasets.dataset.Dataset],
    as explained in [this page][concept-database].

Evaluation problems are fundamental building elements used in more complex problem types
and are commonly used for sensitivity analysis, design space exploration, or validating models against known data points.

[EvaluationProblem][gemseo.algos.evaluation_problem.EvaluationProblem] does with [functions][concept-functions]
what [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario] does with [disciplines][concept-discipline].
Please refer to the [scenario section][concept-scenarios] for more information.

!!! warning

    The methods
    [preprocess_functions()][gemseo.algos.evaluation_problem.EvaluationProblem.preprocess_functions]
    and
    [evaluate_functions()][gemseo.algos.evaluation_problem.EvaluationProblem.evaluate_functions]
    are intended for use by evaluation algorithms.
    They are documented here solely to explain the concepts.
    If you need to manipulate them directly, do so with caution.
