---
reading_time: true
complexity: intermediate
description: "An ODE solver integrates an ODE problem from its initial state over its time interval and stops early when an event function changes sign."
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

An ODE solver integrates an [ODE problem][concept-ode-problem]:
it computes the state from the initial state over the time interval.

## Algorithms { #concept-ode-algorithms }

GEMSEO wraps the ODE solvers of the SciPy function `solve_ivp`.
The [ODE catalog](https://gemseo.org/more_on/catalogs/ode/) lists them with their settings.
The plugin [gemseo-petsc](https://gemseo.gitlab.io/dev/gemseo-petsc) provides other ODE solvers.
It also computes the sensitivities of the solution
with respect to the initial state and the design variables by an adjoint method.
The [plugin catalog](https://gemseo.org/plugins/catalog/) lists the GEMSEO plugins.

??? note "Jacobian of the right-hand side"
    The solvers `Radau`, `BDF` and `LSODA` use the Jacobian
    of the right-hand side $f$ with respect to the state.
    The ODE problem gives it
    as a constant matrix or as a function of the time and the state.
    Otherwise, it is approximated.

!!! tutorial
    - [Tutorial - Solving an ODE with ODEProblem][]
    - [Tutorial - Solving an ODE with GEMSEO][]
    - [Tutorial - Solving a system of coupled ODEs][]

## Result { #concept-ode-result }

The solver stores its result in the `result` attribute of the ODE problem,
an [ODEResult][gemseo.ode.result.ODEResult].
It holds the final state and the termination time,
the numbers of evaluations of $f$ and of its Jacobian,
whether the solver has converged and its termination message.
When the solver has not converged, GEMSEO logs a warning.

??? note "Stored states"
    The solver stores the state at the evaluation times of the ODE problem.
    When an event stops the integration, it stores only the evaluation times before the event.
    If the ODE problem asks to solve at the algorithm times,
    it stores the state at the times chosen by the solver.
    Otherwise, without evaluation times, it stores only the final state,
    but `result.times` still holds the times chosen by the solver.

## Event functions { #concept-ode-event-functions }

An event function is a function of the time and the state.
The integration stops when one of the event functions of the ODE problem changes sign.
Then, the result gives the index of this event function,
the time of this event and the state at this time.
Without event, the solver integrates over the whole time interval.

!!! how-to
    - [Add a termination condition to an ODE][]
