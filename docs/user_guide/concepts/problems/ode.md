---
description: "An ODE problem defines an initial value problem where a numerical solver integrates the state trajectory over a time interval."
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

# ODE problem { #concept-ode-problem }

An ordinary differential equation (ODE) problem involves
finding functions that satisfy a differential equation relating the function to its derivatives.
These problems model dynamic systems that evolve over time or another independent variable,
such as physical processes, chemical reactions, or control systems.
ODE problems require specifying initial conditions or boundary conditions
and typically involve numerical integration methods to compute solutions.
They are essential for simulating time-dependent behavior and analyzing system dynamics.

An initial value problem (IVP) is an ODE together with an initial condition in the form:

$$
\begin{cases}
&\text{Find the function } y(t) \\
&\text{defined on the time interval }[t_0, t_f] \\
&\text{such that }y(t_0) = y_0, \\
&\text{and }\frac{\mathrm{d}}{\mathrm{d}t}y(t) = f(t, y(t))\text{ for all } t \in [t_0, t_f].
\end{cases}
$$

The term $y_0$ identifies the state variable $y$ at the initial time $t_0$.
The right-hand side function $f: t, y \mapsto f(t, y)$ defines the dynamics of the problem,
by computing the time derivative of the state variable $y$ at time $t$.
The function $y(\cdot)$ is computed on the time interval $[t_0, t_f]$
using an algorithm called an ODE solver.

An [ODEProblem][gemseo.algos.ode.ode_problem.ODEProblem] is specified by
the RHS function $f$ as `rhs_function`,
the initial state $y_0$ as `initial_state`,
and the time interval $[t_0, t_f]$ as `time_interval`.
Optionally,
`evaluation_times` lists specific instants within $[t_0, t_f]$
at which the state must be stored, in addition to the final time.

The Jacobian of $f$ with respect to the state,
`jac_function_wrt_state`,
can be provided to accelerate implicit solvers; if absent, it is approximated automatically.
The Jacobian of $f$ with respect to the design variables,
`jac_function_wrt_desvar`,
is needed when propagating sensitivities through the integration.

Sometimes,
we do not want to find the solution $y$ for the entire interval $[t_0, t_f]$,
but only until a stopping criterion is met.
Given a list of termination functions $g_1, \ldots, g_m$
taking as arguments the time $t$ and the state $y$,
the ODE solver stops as soon as one of these functions equals 0.
The ODE problem with termination functions can thus be written as follows:

$$
\begin{cases}
&\text{Find the function } y(t) \\
&\text{defined on the time interval }[t_0, t^*] \\
&\text{such that }y(t_0) = y_0,\\
&\frac{\mathrm{d}}{\mathrm{d}t}y(t) = f(t, y(t))\text{ for all } t \in [t_0, t^*], \\
&\text{and }t^* \text{is the smallest element of }[t_0, t_f] \\
&\text{for which } g_i(t^*, y(t^*)) = 0 \text{ for some } i \in \lbrace 1, \ldots, m\rbrace.
\end{cases}
$$

An ODE solver is typically used to solve an [ODEProblem][gemseo.algos.ode.ode_problem.ODEProblem].
Please refer to the [ODE solvers section][concept-ode-solvers] for more information.

Once solved,
the `result` attribute holds an [ODEResult][gemseo.algos.ode.ode_result.ODEResult]
describing the state trajectories at the evaluation times,
the final state and termination time,
the index of the termination event (if any),
and convergence information from the solver.
