---
reading_time: true
status: draft
description: "GEMSEO integrates ordinary differential equations with a choice of explicit, implicit and backwards-differentiation solvers, and can stop the integration early using event functions."
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

# ODE solvers { #concept-ode-solvers }

## Generalities { #concept-generalities }

GEMSEO integrates first-order ordinary differential equations
by computing the solution of an Initial Value Problem (IVP).
See [ODE problem][concept-ode-problem] for the formal definition of an IVP.

## Algorithms for the numerical solution { #concept-algorithms-for-the-numerical-solution }

Multiple algorithms are available in the literature.
The algorithms available in GEMSEO, developed in the method `solve_ivp`
of the library `scipy.integrate`, are:

* the *explicit Runge-Kutta* algorithms (`RK45`, `RK23`, `DOP853`),
* an *implicit Runge-Kutta* algorithm (`Radau`),
* and two algorithms based on a backwards differentiation formula (`BDF` and `LSODA`).

The algorithms `Radau`, `BDF`, and `LSODA` require the knowledge of the Jacobian of
the function $f$ with respect to the state $y$: $J f = \frac{\partial f}{\partial y}$.
The Jacobian can be either passed to the algorithm, or computed by finite differences.

Further algorithms for the solution of IVPs are available in the plugin
[gemseo-petsc](https://gitlab.com/gemseo/dev/gemseo-petsc).
The plugin [gemseo-petsc](https://gitlab.com/gemseo/dev/gemseo-petsc) provides also an adjoint mode
to perform a sensitivity analysis on the solution of the ODE
with respect to its initial values and the design parameters.

## Event functions { #concept-event-functions }

For some problems, it might be interesting not to integrate the dynamic for the entire
time interval $[t_0, t_f]$, but only up to the realization of a terminating condition.
Such conditions are encoded by **event functions**:
real-valued continuous functions $g_1, \ldots, g_m: [t_0, t_f] \times \mathbb{R}^n \rightarrow \mathbb{R}$.
The terminating condition is realized when any of the event function crosses the threshold $0$.
