---
reading_time: true
complexity: intermediate
description: "An ODEDiscipline wraps an ODEProblem as a Discipline, allowing ODE dynamics to be coupled with the rest of an MDO process."
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

# ODE discipline { #concept-ode-discipline }

An [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline] is the subclass of [Discipline][gemseo.core.discipline.discipline.Discipline] wrapping an [ODEProblem][gemseo.ode.problem.ODEProblem].

The function $f(t, y)$ defining the right-hand side of the ODE and the termination functions are encoded by
instances of [Discipline][gemseo.core.discipline.discipline.Discipline] with suitable inputs and outputs, allowing to couple different instances of
[ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline] in an [MDA][concept-solving-multi-disciplinary-analysis].

## Inputs and outputs

An instance of [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline] takes as inputs:

* the initial value of the *time* variable,
* the initial value of the *state* variables,
* the value of eventual *design variables*.

Without further specifications, the outputs of [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline] are the values of the state variables at the end
of the time interval (or, if *termination events* are present, at the realization of the first event).
By default, the name of the output variable corresponding to the final value of the state variable `"y"` is `"y_final"`.

If the `return_trajectories` parameter is set to `True`, the discipline additionally outputs the state values at the
instants listed in `times`; the trajectory output for state `"y"` is named `"y"`.

## Initialization

The instantiation of an [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline] requires at least two parameters: `discipline`, representing the
function $f(t, y)$, and `times`, representing the time interval of integration of the ODE.
Further parameters can be specified at the time of the instantiation of the [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline].
