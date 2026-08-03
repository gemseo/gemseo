# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""# Tutorial - Solving an ODE with GEMSEO

## Goal

This tutorial introduces the
[ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline],
the GEMSEO component for solving first-order ordinary differential equations (ODEs).

A first-order ODE is a differential equation of the form

$$\\frac{ds(t)}{dt} = f(t, s(t))$$

where $f$ is the right-hand side (RHS) function,
$t$ is time, and $s(t)$ is the state vector.
Solving it requires initial conditions $s(t_0) = s_0$.

You will learn how to:

- **define** the RHS of an ODE as a GEMSEO discipline,
- **create** an [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline]
  from the RHS discipline,
- **execute** it with default and custom inputs,
- **compare** the numerical solution with the analytical one.

As a running example, this tutorial uses the harmonic oscillator:

$$\\frac{d^2 x(t)}{dt^2} + \\omega^2 x(t) = 0$$

Rewritten as a first-order system by introducing $y = \\frac{dx}{dt}$
and the state vector $s = \\begin{pmatrix}x\\\\y\\end{pmatrix}$:

$$\\frac{ds(t)}{dt} = \\begin{pmatrix} y(t) \\\\ -\\omega^2 x(t) \\end{pmatrix}$$

The analytical solution with initial position $x_0$ and velocity $v_0$ is:

$$x(t) = x_0 \\cos(\\omega t) + \\frac{v_0}{\\omega} \\sin(\\omega t)$$

!!!quote "References"
         Philip Hartman.
         Ordinary differential equations.
         SIAM, 2002.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import array
from numpy import cos
from numpy import linspace
from numpy import ndarray  # noqa: TC002
from numpy import pi
from numpy import sin

from gemseo import create_discipline
from gemseo.core.discipline.discipline import Discipline
from gemseo.discipline import ODEDiscipline

# %%
# ## Step 1 — Define the RHS discipline
#
# The RHS discipline represents the function $f(t, s(t))$.
# Its inputs must include the time variable and all state variables;
# any other input is treated as a design variable (here `omega`).
# Its outputs must be the time derivatives of the state variables,
# named with the `_dot` suffix by convention.
#
# You use an [AutoPyDiscipline][gemseo.discipline.auto_py.AutoPyDiscipline]
# built from a plain Python function:
_time = array([0.0])
initial_position_1 = array([1.0])
initial_velocity_1 = array([0.0])
omega_1 = array([2.0])


def rhs_function(
    time: ndarray = _time,
    position: ndarray = initial_position_1,
    velocity: ndarray = initial_velocity_1,
    omega: ndarray = omega_1,
) -> tuple[ndarray, ndarray]:
    position_dot = velocity
    velocity_dot = -(omega**2) * position
    return position_dot, velocity_dot


rhs_discipline = create_discipline(
    "AutoPyDiscipline",
    py_func=rhs_function,
    grammar_type=Discipline.GrammarType.SIMPLE,
)

# %%
# ## Step 2 — Create the ODEDiscipline
#
# The [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline]
# wraps the RHS discipline into a complete initial-value problem.
#
# - `rhs_discipline`: the discipline representing the right-hand side $f(t, s(t))$.
# - `times`: the time grid, with at least the initial and final times.
# - `state_names`: the names of the state variables, to distinguish them
#   from the time variable and the design variables.
# - `return_trajectories=True`: if `True`, the full state trajectory is returned in the output,
#   not just the final state.
ode_discipline = ODEDiscipline(
    rhs_discipline=rhs_discipline,
    times=linspace(0.0, 10.0, 51),
    state_names=["position", "velocity"],
    return_trajectories=True,
)

# %%
# ## Step 3 — Execute with default inputs
#
# The default inputs of the
# [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline]
# are inherited from the RHS discipline.
# Calling `execute()` without arguments solves the ODE
# with $\omega = 2$, $x_0 = 1$ and $v_0 = 0$:
ode_res_1 = ode_discipline.execute()

# %%
# ## Step 4 — Execute with custom inputs
#
# Different initial conditions and design variables can be passed
# as a dictionary to `execute()`.
# Here you solve the problem with $\omega = 1$, $x_0 = 2$ and $v_0 = 0.5$:
initial_position_2 = array([2.0])
initial_velocity_2 = array([0.5])
omega_2 = array([1.0])

ode_res_2 = ode_discipline.execute({
    "initial_position": initial_position_2,
    "initial_velocity": initial_velocity_2,
    "omega": omega_2,
})

# %%
# ## Step 5 — Compare with the analytical solution
#
# You verify the numerical solutions against the analytical formula
# $x(t) = x_0 \cos(\omega t) + \frac{v_0}{\omega} \sin(\omega t)$.
#
# First solution ($\omega = 2$, $x_0 = 1$, $v_0 = 0$):
analytic_res_1 = initial_position_1 * cos(omega_1 * ode_res_1["times"]) + (
    initial_velocity_1 / omega_1
) * sin(omega_1 * ode_res_1["times"])

plt.plot(ode_res_1["times"], analytic_res_1, "r", label="Analytical solution")
plt.plot(ode_res_1["times"], ode_res_1["position"], "b--", label="ODEDiscipline")
plt.legend()
frequency = omega_1[0] / (2 * pi)
plt.title(f"Harmonic oscillator with frequency {omega_1[0]}/(2π) = {frequency:.3f}")
plt.show()

# %%
# Second solution ($\omega = 1$, $x_0 = 2$, $v_0 = 0.5$):
analytic_res_2 = initial_position_2 * cos(omega_2 * ode_res_2["times"]) + (
    initial_velocity_2 / omega_2
) * sin(omega_2 * ode_res_2["times"])

plt.plot(ode_res_2["times"], analytic_res_2, "r", label="Analytical solution")
plt.plot(ode_res_2["times"], ode_res_2["position"], "b--", label="ODEDiscipline")
plt.legend()
frequency = omega_2[0] / (2 * pi)
plt.title(f"Harmonic oscillator with frequency {omega_2[0]}/(2π) = {frequency:.3f}")
plt.show()

# %%
# ## Key takeaways
#
# - An [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline]
#   solves a first-order ODE initial-value problem defined by a RHS discipline.
#   The RHS discipline must output the time derivatives of the state variables.
# - Use `state_names` to tell the
#   [ODEDiscipline][gemseo.discipline.ode.ode_discipline.ODEDiscipline]
#   which inputs are state variables;
#   all other non-time inputs are treated as design variables.
# - Different initial conditions and design variable values can be passed
#   at execution time via a dictionary, without recreating the discipline.
# - Set `return_trajectories=True` to retrieve the full state trajectory
#   over the time grid, not just the final state.
#
# !!! note
#
#     The [OscillatorDiscipline][gemseo.problem.ode.oscillator_discipline.OscillatorDiscipline]
#     class provides a ready-to-use implementation of the harmonic oscillator.
#     It can be imported from
#     [gemseo.problem.ode.oscillator_discipline][gemseo.problem.ode.oscillator_discipline]
#     and used directly without defining the RHS function manually.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Add a termination condition to an ODE][add-a-termination-condition-to-an-ode]
