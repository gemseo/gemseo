<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Optimizing a design { #usecases-optimizing-a-design }

## What it means { #usecases-what-it-means }

Optimization seeks the **best design**---the
set of input values that minimizes (or maximizes) a performance metric
while satisfying constraints.

In mathematical terms, the optimization problem reads:

$$
\min_{x \in \mathcal{X}} f(x)
\quad \text{subject to} \quad
g(x) \leq 0, \quad
h(x) = 0
$$

where $f$ is the objective function,
$g$ and $h$ are inequality and equality constraints,
and $\mathcal{X}$ is the design space defined by variable bounds and types.

## Why it matters { #usecases-why-it-matters }

Design optimization goes beyond manual trial-and-error
or DOE-based exploration:
it **systematically searches** for the best solution,
potentially in high-dimensional spaces
where human intuition is insufficient.

In an engineering context,
optimization can reduce weight, cost, or fuel consumption,
improve performance or reliability,
or find compromises between conflicting objectives.

## How GEMSEO does it { #usecases-how-gemseo-does-it }

In GEMSEO, optimization is orchestrated by a **scenario**.
The user defines:

1. The **disciplines** involved.
2. The **design space**: variables, bounds, types (continuous or integer).
3. The **objective function** and **constraints** (chosen among discipline outputs).
4. The **MDO formulation** (e.g. MDF, IDF) which defines how disciplines are coupled during optimization.
5. The **optimization algorithm** and its settings.

GEMSEO then builds the process,
translates the user's problem description
into a mathematical `OptimizationProblem`,
and hands it to the chosen algorithm.

### Available optimization algorithms { #usecases-available-optimization-algorithms }

GEMSEO wraps 30+ optimization algorithms, including:

- **Gradient-based**: L-BFGS-B, SLSQP, SNOPT (via pyOptSparse)---fast convergence when derivatives are available.
- **Gradient-free**: COBYLA, Nelder-Mead, differential evolution, CMA-ES---suitable when derivatives are not available or the objective is noisy.
- **Multi-objective**: NSGA-II, NSGA-III---for Pareto front generation when multiple objectives must be balanced.
- **Mixed-integer**: algorithms supporting both continuous and discrete variables.

Derivatives can be provided analytically by the disciplines,
or approximated by GEMSEO
using finite differences or the complex step method.
For coupled systems,
GEMSEO computes **coupled derivatives** (direct or adjoint methods)
through the MDA.

### Post-processing { #usecases-post-processing }

After optimization,
GEMSEO provides a rich set of post-processing tools:

- Objective and constraint convergence history.
- Design variable evolution.
- Sensitivity analysis at the optimum.
- Pareto front visualization (multi-objective).
- Robustness analysis of the optimum under design variable uncertainty.
- Constraint radar charts and self-organizing maps.
