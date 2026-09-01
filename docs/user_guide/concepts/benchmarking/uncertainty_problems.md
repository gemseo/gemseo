---
reading_time: true
complexity: advanced
description: "Uncertainty quantification problems with known analytical statistics bundled with GEMSEO for benchmarking sensitivity analysis and UQ algorithms."
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

# Uncertainty problems { #concept-uncertainty-problems }

The [gemseo.problem.uncertainty][gemseo.problem.uncertainty] package
provides problems with closed-form statistics
for benchmarking uncertainty quantification and global sensitivity analysis algorithms.
Estimated quantities can be compared directly against analytical reference values.

## Ishigami { #concept-ishigami }

The Ishigami function is a standard benchmark
for global sensitivity analysis [@ishigami1990]:

$$f(x_1, x_2, x_3) = \sin(x_1)\,(1 + 0.1\,x_3^4) + 7\,\sin^2(x_2)$$

where $x_1, x_2, x_3 \sim \mathcal{U}[-\pi, \pi]$ are independent uniform random variables.

### Analytical statistics { #concept-ishigami-statistics }

The following exact values are available for validation:

| Quantity | Value |
|----------|-------|
| Mean | $3.5$ |
| Variance | $\approx 13.845$ |
| First-order Sobol' index $S_1$ | $\approx 0.5576$ |
| First-order Sobol' index $S_2$ | $\approx 0.2442$ |
| First-order Sobol' index $S_3$ | $0$ |
| Total Sobol' index $T_1$ | $\approx 0.5576$ |
| Total Sobol' index $T_2$ | $\approx 0.2442$ |
| Total Sobol' index $T_3$ | $\approx 0.2442$ |

Variable $x_3$ has zero first-order effect but a non-zero total effect
due to the interaction with $x_1$, making the Ishigami function a useful
test case for detecting higher-order interactions.

### API { #concept-ishigami-api }

The Ishigami problem is available in four forms:

| Class | Base class | Use case |
|-------|------------|----------|
| [IshigamiDiscipline][gemseo.problem.uncertainty.ishigami.ishigami_discipline.IshigamiDiscipline] | [Discipline][gemseo.core.discipline.discipline.Discipline] | MDO integration, coupling |
| [IshigamiFunction][gemseo.problem.uncertainty.ishigami.ishigami_function.IshigamiFunction] | [ArrayFunction][gemseo.core.function.array_function.ArrayFunction] | Low-level function evaluation |
| [IshigamiProblem][gemseo.problem.uncertainty.ishigami.ishigami_problem.IshigamiProblem] | [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem] | Uncertainty-aware optimization |
| [IshigamiSpace][gemseo.problem.uncertainty.ishigami.ishigami_space.IshigamiSpace] | [ParameterSpace][gemseo.space.parameter.ParameterSpace] | Probabilistic input space |

The [statistics][gemseo.problem.uncertainty.ishigami.statistics] module
exports all analytical reference values as constants
(`MEAN`, `VARIANCE`, `SOBOL_1`, `SOBOL_2`, `SOBOL_3`, `TOTAL_SOBOL_1`, etc.)
for direct use in test assertions.
