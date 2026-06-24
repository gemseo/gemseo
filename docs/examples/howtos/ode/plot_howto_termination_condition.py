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
"""# Add a termination condition to an ODE

## Problem

By default, an
[ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline]
solves the ODE over the entire time interval.
You want the integration to stop early
when a specific condition is met - for example,
when a state variable crosses a given threshold.

## Solution

Pass a list of termination condition disciplines to the
`termination_event_disciplines` argument of
[ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline].
Each discipline must have the same inputs as the RHS discipline
and return a single real-valued output.
The integration stops when that output crosses zero.

## Step-by-step guide
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import array
from numpy import cos
from numpy import linspace
from numpy import ndarray  # noqa: TC002
from numpy import sin

from gemseo import create_discipline
from gemseo.core.discipline.discipline import Discipline
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline

# %%
# ### 1. Build the RHS discipline
#
# You reuse the harmonic oscillator RHS from the ODE tutorial.
#
# !!! tutorial
#     - [tutorial - Solving an ODE with GEMSEO][tutorial-solving-an-ode-with-gemseo]
_time = array([0.0])
initial_position = array([1.5])
initial_velocity = array([0.0])
omega = array([2.0])


def rhs_function(
    time: ndarray = _time,
    position: ndarray = initial_position,
    velocity: ndarray = initial_velocity,
    omega: ndarray = omega,
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
# ### 2. Define the termination condition discipline
#
# The termination discipline must have the same inputs as the RHS discipline
# and return a single real-valued output.
# The integration stops when this output crosses zero.
# Here you stop when the oscillator reaches the equilibrium position ($x = 0$):
def termination_function(
    time: ndarray = _time,
    position: ndarray = initial_position,
    velocity: ndarray = initial_velocity,
    omega: ndarray = omega,
) -> ndarray:
    termination = position
    return termination


termination_discipline = create_discipline(
    "AutoPyDiscipline",
    py_func=termination_function,
    grammar_type=Discipline.GrammarType.SIMPLE,
)

# %%
# ### 3. Create the ODEDiscipline with the termination condition
#
# Pass the termination discipline as a tuple
# to `termination_event_disciplines`:
ode_discipline = ODEDiscipline(
    rhs_discipline=rhs_discipline,
    times=linspace(0.0, 10.0, 51),
    state_names=["position", "velocity"],
    termination_event_disciplines=(termination_discipline,),
    return_trajectories=True,
    solve_at_algorithm_times=True,
)

# %%
# ### 4. Execute and inspect the result
#
# The integration stops as soon as `position` crosses zero:
ode_res = ode_discipline.execute({
    "initial_position": initial_position,
    "initial_velocity": initial_velocity,
    "omega": omega,
})

analytic_res = initial_position * cos(omega * ode_res["times"]) + (
    initial_velocity / omega
) * sin(omega * ode_res["times"])

plt.plot(ode_res["times"], analytic_res, "r", label="Analytical solution")
plt.plot(ode_res["times"], ode_res["position"], "b--", label="ODEDiscipline")
plt.plot(
    ode_res["times"],
    [0.0] * len(ode_res["times"]),
    "k--",
    label="Termination threshold = 0.0",
)
plt.legend()
plt.title("Harmonic oscillator with termination condition")
plt.show()

# %%
# ## Summary
#
# Pass one or more termination condition disciplines to
# `termination_event_disciplines` when creating an
# [ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline].
# Each discipline must share the same inputs as the RHS discipline
# and return a single real value; the integration stops when it crosses zero.
