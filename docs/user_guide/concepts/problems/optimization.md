---
description: "An optimization problem extends an evaluation problem with an objective function to minimize or maximize and optional constraints."
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

# Optimization problem { #concept-optimization-problem }

An [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
extends an [EvaluationProblem][gemseo.algos.evaluation_problem.EvaluationProblem]
by introducing an *objective* function to minimize or maximize
and optional *constraints* that solutions must satisfy.
It operates on a [DesignSpace][gemseo.algos.design_space.DesignSpace]
defining the design variables $x$,
their bounds $m \leq x \leq M$,
and an initial guess $x_0$.
In its most general form:

$$\min_{x} f(x) \quad \text{s.t.} \quad g(x) \leq 0,\quad h(x) = 0,\quad m \leq x \leq M$$

Since this is an evaluation problem extension,
it may also involve observable-type functions.

The objective is an [ArrayFunction][gemseo.core.functions.array_function.ArrayFunction]
set via the `objective` attribute.
By default the problem minimizes $f(x)$;
setting `minimize_objective` to `False` switches to maximization
by internally negating the function so that optimizers always minimize.

Equality ($h(x) = a$) and inequality ($g(x) \leq a$ or $g(x) \geq a$) constraints
are [ArrayFunction][gemseo.core.functions.array_function.ArrayFunction] instances
added via [add_constraint()][gemseo.algos.optimization_problem.OptimizationProblem.add_constraint].
They are internally rewritten in the general form mentioned above.
Feasibility is checked against configurable
[ConstraintTolerances][gemseo.algos.constraint_tolerances.ConstraintTolerances]
accessible through the `tolerances` attribute.

An optimization algorithm is typically used to solve an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem],
even though a [design of experiments (DOE)][concept-samplers-doe] can be used for a preliminary trade-off analysis.
Please refer to the [optimizers section][concept-optimizers] for more information.

Once solved,
the `solution` attribute holds an [OptimizationResult][gemseo.algos.optimization_result.OptimizationResult]
describing the optimal design, objective value, constraint values and feasibility.
The `optimum` property returns the best feasible point found over the full optimization history.

The [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
with its [Database][gemseo.algos.database.Database] can be exported to an HDF file
using [to_hdf()][gemseo.algos.optimization_problem.OptimizationProblem.to_hdf],
and restored using [from_hdf()][gemseo.algos.optimization_problem.OptimizationProblem.from_hdf].

[OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] does with [functions][concept-functions]
what [MDOScenario][gemseo.scenarios.mdo.MDOScenario] does with [disciplines][concept-discipline].
Please refer to the [scenario section][concept-scenarios] for more information.
