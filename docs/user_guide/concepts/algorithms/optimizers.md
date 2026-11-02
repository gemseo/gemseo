---
reading_time: true
complexity: intermediate
description: "An optimizer searches for the best point of an optimization problem, iteratively or by sampling, and returns the best feasible point it evaluated."
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

# Optimizers { #concept-optimizers }

An optimizer searches for the best point
of an [optimization problem][concept-optimization-problem]:

$$
\begin{aligned}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & g(x) \leq 0, \\
& h(x) = 0, \\
& m \leq x \leq M
\end{aligned}
$$

where $f$ is the objective,
$g$ and $h$ are the inequality and equality constraints
and $m$ and $M$ are the bounds of the design variables $x$.

The objective is mandatory:
a problem without objective cannot be solved.
The constraints are optional.
To maximize the objective,
the problem minimizes its opposite,
so that the optimizers always minimize.

An optimizer is selected and configured by its [settings][settings-user-guide].

## Two ways to solve a problem { #concept-optimizer-two-ways }

An optimization algorithm is iterative.
It starts from an initial point $x_0$,
the current value of the [design space][concept-design-space].
The missing current values are initialized automatically,
according to the rules of the [variable types][concept-variable-types].
At each iteration,
the algorithm evaluates the functions,
and their gradients if it needs them,
then proposes a new point.

A [design of experiments (DOE)][concept-samplers-doe] can also process an optimization problem.
It evaluates the functions at points chosen before the evaluations.
Applied to an optimization problem,
it returns an optimization result too:
the best of these points.
It is useful for a preliminary trade-off analysis.

In both cases,
the evaluations are stored in the [database][concept-database] of the problem.

!!! how-to
    - [Record the evaluations in the coordinates of the algorithm][]
    - [Evaluate the functions outside the bounds of the design space][]

## What an algorithm can handle { #concept-algorithm-categories }

Not every optimization algorithm can solve every optimization problem.
Each algorithm declares what it can handle:

- the gradient:
  a gradient-based algorithm requires the gradients of the functions;
- the constraints:
  an algorithm can handle equality constraints,
  inequality constraints,
  both or none;
- the linear problems:
  an algorithm can be dedicated to linear problems,
  where the objective and the constraints are linear;
- the multiple objectives,
  see [Multiple objectives][concept-optimizer-multiple-objectives].

A problem is rejected
when it has constraints that the algorithm does not handle
or when it is non-linear while the algorithm is dedicated to linear problems.
The algorithm also declares the types of [variables][concept-variable-types] it handles.

??? note "Attributes of the description"
    The
    [OptimizationAlgorithmDescription][gemseo.optimization.core.base_optimization_library.OptimizationAlgorithmDescription]
    of an algorithm records what it can handle:

    - `require_gradient` for the gradient,
    - `handle_equality_constraints` and `handle_inequality_constraints` for the constraints,
    - `for_linear_problems` for the linear problems,
    - `handle_multiobjective` for the multiple objectives.

!!! how-to
    - [Assess the constraints at a point][]
    - [Transform a constrained problem into an unconstrained one][]
    - [How to aggregate constraints][]

!!! tutorial
    - [Tutorial - Solve a mixed optimization problem][]

## Multiple objectives { #concept-optimizer-multiple-objectives }

The objective can have several components.
The problem is then multi-objective.
Its solution is not a single point
but a Pareto front:
the set of points
where no component can be improved without degrading another one.

An [MDO scenario][gemseo.scenario.mdo.MDOScenario] can have several objectives,
some to minimize and others to maximize,
see [add_objective()][gemseo.scenario.mdo.MDOScenario.add_objective].
The objectives to maximize are negated,
so that the problem only minimizes.

The MNBI algorithm computes this Pareto front
by decomposing the problem into a series of constrained single-objective problems.
Its result includes the Pareto front
when the solution is feasible.

!!! how-to
    - [Use the mNBI algorithm][]

## Local and global algorithms { #concept-optimizer-local-global }

A local algorithm searches for an optimum
in the neighborhood of its starting point.
It finds a local optimum:
a point better than its neighbors,
not necessarily the best point of the design space.
The gradient-based algorithms are local.

A global algorithm explores the whole design space
to find the global optimum,
the best point of the design space.
It usually needs many more evaluations of the functions.
SciPy offers global algorithms,
such as differential evolution.

## Multi-start optimization { #concept-multi-start-optimization }

The MultiStart algorithm searches for a global optimum with a local algorithm:
it generates starting points with a DOE algorithm
and runs a sub-optimization from each of them.
The solution is the best feasible point evaluated by all the sub-optimizations.

??? note "Settings"
    The DOE algorithm is LHS by default
    and is configured by `doe_algo_settings`.
    The sub-optimization algorithm is SLSQP by default
    and is configured by `opt_algo_settings`.

    The sub-optimizations share the maximum number of iterations `max_iter`.
    When the field `max_iter` of `opt_algo_settings` is not set,
    MultiStart splits `max_iter` evenly among the sub-optimizations.
    Otherwise,
    the sum of the maximum numbers of iterations of the sub-optimizations
    must not exceed `max_iter`.

    `n_processes` is the maximum number of processes
    used to parallelize the sub-optimizations.
    `multistart_file_path` is the database file path to save the local optima.
    `skip_failed_starting_points` tells whether to skip the starting points
    whose sub-optimization fails.
    In this case,
    an error is raised only when all the starting points fail.

!!! how-to
    - [Find a global optimum with a local algorithm][]

## Stopping criteria { #concept-optimizer-stopping-criteria }

Besides the stopping criteria of the algorithm,
GEMSEO applies criteria common to all the algorithms.
It stops the algorithm when:

- the maximum number of iterations is reached,
- the maximum runtime is reached,
- the objective stagnates over the last iterations,
- the design variables stagnate over the last iterations,
- the residual of the Karush-Kuhn-Tucker (KKT) conditions is small enough,
  for the gradient-based algorithms only.

The two stagnation criteria can be met
only when at least one of the last iterations is feasible.
The computation of the KKT residual is inspired
by the implementation of the MMA algorithm by Svanberg [@svanberg1998method].

When GEMSEO stops the algorithm,
the message of the optimization result ends with "GEMSEO stopped the driver.".

??? note "Settings of the stopping criteria"
    The settings are defined in
    [BaseOptimizerSettings][gemseo.optimization.core.base_optimizer_settings.BaseOptimizerSettings].
    Some libraries change their default values.

    - `max_iter` is the maximum number of iterations,
      1000 by default.
    - `max_time` is the maximum runtime in seconds,
      disabled if 0.
    - `ftol_rel` and `ftol_abs` are the relative and absolute tolerances
      on the objective.
    - `xtol_rel` and `xtol_abs` are the relative and absolute tolerances
      on the design variables.
      These four tolerances are 0 by default,
      which disables the stagnation criteria.
    - `stop_crit_n_x` is the number of last iterations
      considered by the stagnation criteria,
      3 by default.
    - `kkt_tol_abs` and `kkt_tol_rel` are the absolute and relative tolerances
      on the norm of the KKT residual.
      The criterion is not activated when both tolerances are infinite,
      which is the default.
      It requires `store_jacobian` to be `True`.

## NaN values { #concept-optimizer-nan-values }

By default,
a NaN in the value or in the gradient of a function stops the optimizer.
A NaN in the design variables stops it too.
The result is then built from the points already evaluated.

Setting the attribute `stop_if_nan` of the problem to `False` disables this behavior.

A point whose constraint value contains NaN
or is missing
is infeasible.

## Result { #concept-optimizer-result }

The result is an
[OptimizationResult][gemseo.optimization.result.OptimizationResult],
stored in the attribute `solution` of the problem.
It is built from the database of the problem,
even when the optimizer is stopped before its end.
So the optimum is not the last point evaluated by the algorithm
but the best point of the database:

- the feasible point with the lowest objective value,
- or, if no point is feasible,
  the least infeasible point,
  with a warning.

??? note "Content of the result"
    An [OptimizationResult][gemseo.optimization.result.OptimizationResult]
    includes:

    - the initial and optimal values of the design variables,
    - the objective value at the optimum,
    - whether the optimum is feasible,
    - the values and gradients of the constraints at the optimum,
    - the numbers of calls to the objective, its gradient and the constraints,
    - the position of the optimum in the optimization history,
    - the name of the optimizer,
    - the message and the status returned by the optimizer.

## Wrapped libraries { #concept-wrapped-libraries }

GEMSEO wraps optimization algorithms of third-party libraries,
such as SciPy and NLopt.
Plugins wrap other libraries.
GEMSEO also implements its own algorithms,
such as the augmented Lagrangian,
MNBI
and MultiStart.

The settings models of the algorithms are in the
[`gemseo.optimization`][gemseo.optimization] package.

!!! note "Available algorithms"
    The [optimization catalog](https://gemseo.org/more_on/catalogs/optimization/)
    lists the available algorithms,
    including the ones of the plugins.
