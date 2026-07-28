---
reading_time: true
description: "Single-objective optimization problems bundled with GEMSEO for benchmarking and illustrating optimization algorithms."
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

# Optimization problems { #concept-optimization-problems }

The [gemseo.problems.optimization][gemseo.problems.optimization] package provides single-objective
[OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] instances
for benchmarking and illustrating optimization algorithms.
All problems include analytical Jacobians.

## Rosenbrock { #concept-rosenbrock }

The Rosenbrock function is a classic non-convex benchmark:

$$f(x) = \sum_{i=1}^{n-1} \left[100\,(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right]$$

It has a global minimum at $x = (1, \ldots, 1)$ with $f = 0$,
located inside a long, narrow, parabolic-shaped flat valley.
The default domain is $[-2, 2]^n$ with $n = 2$.

??? abstract "API"

    - [Rosenbrock][gemseo.problems.optimization.rosenbrock.Rosenbrock]

## Multi-fidelity Rosenbrock { #concept-rosen-mf }

[RosenMF][gemseo.problems.optimization.rosen_mf.RosenMF] is a
[Discipline][gemseo.core.discipline.discipline.Discipline]
that wraps the Rosenbrock function with a fidelity level:

$$y = \text{fidelity} \times f_{\text{Rosenbrock}}(x)$$

where $f_{\text{Rosenbrock}}$ is the standard $n$-dimensional Rosenbrock function.
Setting `fidelity = 1` recovers the exact function.
This discipline is intended for benchmarking multi-fidelity optimization and surrogate
strategies that combine cheap low-fidelity evaluations with expensive high-fidelity ones.

??? abstract "API"

    - [RosenMF][gemseo.problems.optimization.rosen_mf.RosenMF]

## Rastrigin { #concept-rastrigin }

The Rastrigin function is a highly multimodal benchmark:

$$f(x) = 10n + \sum_{i=1}^{n} \left[x_i^2 - 10\cos(2\pi x_i)\right]$$

The global minimum is at $x = 0$ with $f = 0$, surrounded by a large number of local
minima that are regularly distributed.
The problem is defined on $[-0.1, 0.1]^2$ by default.

??? abstract "API"

    - [Rastrigin][gemseo.problems.optimization.rastrigin.Rastrigin]

## Power2 { #concept-power2 }

Power2 is a constrained quadratic problem:

$$\min_{x \in \mathbb{R}^3} \; x_0^2 + x_1^2 + x_2^2$$

subject to

$$x_0^3 - 0.5 \geq 0, \quad x_1^3 - 0.5 \geq 0, \quad x_2^3 - 0.9 = 0.$$

The analytical optimum is $x^* = (0.5^{1/3},\, 0.5^{1/3},\, 0.9^{1/3})$.

??? abstract "API"

    - [Power2][gemseo.problems.optimization.power_2.Power2]

## Hock-Schittkowski 71 { #concept-hs71 }

Problem 71 from Hock & Schittkowski (1981) is a nonlinear programming benchmark:

$$\min_{x \in [1,5]^4} \; x_1\,x_4\,(x_1 + x_2 + x_3) + x_3$$

subject to

$$x_1\,x_2\,x_3\,x_4 \geq 25, \quad x_1^2 + x_2^2 + x_3^2 + x_4^2 = 40.$$

The analytical optimum is known and can be retrieved via `get_solution()`.

??? abstract "API"

    - [HockSchittkowski71][gemseo.problems.optimization.hock_schittkowski_71.HockSchittkowski71]
