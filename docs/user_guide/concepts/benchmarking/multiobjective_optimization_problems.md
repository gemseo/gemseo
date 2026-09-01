---
reading_time: true
complexity: intermediate
description: "Multi-objective optimization problems bundled with GEMSEO for benchmarking and illustrating Pareto front algorithms."
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

# Multi-objective optimization problems { #concept-multiobjective-optimization-problems }

The [gemseo.problem.multiobjective_optimization][gemseo.problem.multiobjective_optimization] package provides
[OptimizationProblem][gemseo.optimization.problem.OptimizationProblem] instances
with multiple conflicting objectives for benchmarking and illustrating
Pareto front approximation algorithms.
All problems include analytical Jacobians.

## Binh-Korn { #concept-binh-korn }

The Binh-Korn problem is a 2-objective, 2-variable problem:

$$\min_{x \in [0,5],\, y \in [0,3]} \; \bigl(4x^2 + 4y^2,\; (x-5)^2 + (y-5)^2\bigr)$$

subject to

$$(x-5)^2 + y^2 \leq 25, \quad (x-8)^2 + (y+3)^2 \geq 7.7.$$

??? abstract "API"

    - [BinhKorn][gemseo.problem.multiobjective_optimization.binh_korn.BinhKorn]

## Fonseca-Fleming { #concept-fonseca-fleming }

The Fonseca-Fleming problem is a bi-objective problem of configurable dimension $d$:

$$f_1(x) = 1 - \exp\!\left(-\sum_{i=1}^{d}\!\left(x_i - \tfrac{1}{\sqrt{d}}\right)^{\!2}\right), \quad
  f_2(x) = 1 + \exp\!\left(-\sum_{i=1}^{d}\!\left(x_i + \tfrac{1}{\sqrt{d}}\right)^{\!2}\right)$$

with $x \in [-4, 4]^d$.
The two objectives are minimized simultaneously; the Pareto front is a 1D curve.

??? abstract "API"

    - [FonsecaFleming][gemseo.problem.multiobjective_optimization.fonseca_fleming.FonsecaFleming]

## Poloni { #concept-poloni }

The Poloni problem is a 2-objective, 2-variable benchmark defined on $[-\pi, \pi]^2$:

$$f_1(x, y) = (x + 3)^2 + (y + 1)^2$$

$$f_2(x, y) = 1 + (A_1 - B_1)^2 + (A_2 - B_2)^2$$

where $A_1 = 0.5\sin(1) - 2\cos(1) + \sin(2) - 1.5\cos(2)$,
$A_2 = 1.5\sin(1) - \cos(1) + 2\sin(2) - 0.5\cos(2)$,
$B_1 = 0.5\sin(x) - 2\cos(x) + \sin(y) - 1.5\cos(y)$,
$B_2 = 1.5\sin(x) - \cos(x) + 2\sin(y) - 0.5\cos(y)$.

??? abstract "API"

    - [Poloni][gemseo.problem.multiobjective_optimization.poloni.Poloni]

## Viennet { #concept-viennet }

The Viennet problem is a 3-objective, 2-variable benchmark defined on $[-3, 3]^2$:

$$f_1(x, y) = \frac{x^2 + y^2}{2} + \sin(x^2 + y^2)$$

$$f_2(x, y) = \frac{(3x - 2y + 4)^2}{8} + \frac{(x - y + 1)^2}{27} + 15$$

$$f_3(x, y) = \frac{1}{x^2 + y^2 + 1} - 1.1\,\exp\!\bigl(-(x^2 + y^2)\bigr)$$

??? abstract "API"

    - [Viennet][gemseo.problem.multiobjective_optimization.viennet.Viennet]
