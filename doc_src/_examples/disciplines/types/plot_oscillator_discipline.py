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
# Contributors:
# Isabelle Santos
"""
Create a discipline that solves an ODE
======================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import array
from numpy import linspace
from numpy import ndarray

from gemseo import MDODiscipline
from gemseo import create_discipline
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.problems.ode.oscillator_discipline import OscillatorDiscipline

# %%
# This tutorial describes how to use an :class:`.ODEDiscipline`.
#
# An :class:`.ODEDiscipline` is an :class:`.MDODiscipline`
# that solves an ordinary differential equation (ODE).
#
# To illustrate the basic usage of this feature, we use a simple oscillator problem.
#
# Creating an oscillator discipline
# ---------------------------------
#
# Step 1: The ODE describing the motion of the oscillator
# .......................................................
#
# The motion of a simple oscillator is described by the equation
#
# ..math::
#
#     \frac{d^2x(t)}{dt^2} = -\omega^2x(t)
#
# with :math:`\omega \in \mathbb{R}_+^*`.
# As |g| cannot solve a second-order ODE,
# let's re-write this equation as a first-order ODE:
#
# ..math::
#
#     \left\{\begin{array}
#     \frac{dx(t)}{dt} = v(t) \\
#     \frac{dv(t)}{dt} = -\omega^2 x(t)
#     \end{array}\right.
#
# where :math:`x` is the position and :math:`v` is the velocity of the oscillator.
#
# Then,
# we can define the right-hand side (RHS) function
# :math:`(t,x(t),v(t))\mapsto (v(t),-\omega^2x(t))`
# as follows:

omega = 4

time_init = array([0.0])
position_init = array([0.0])
velocity_init = array([1.0])


def compute_rhs_function(
    time: ndarray = time_init,
    position: ndarray = position_init,
    velocity: ndarray = velocity_init,
) -> tuple[ndarray, ndarray]:
    """Evaluate the RHS function :math:`f` of the equation.

    Args:
        time: The time for which :math:`f` should be evaluated.
        position: The position for which :math:`f` should be evaluated.
        velocity: The velocity for which :math:`f` should be evaluated.

    Returns:
        The value of :math:`f` at `time`, `position` and `velocity`.
    """
    position_dot = velocity
    velocity_dot = -(omega**2) * position
    return position_dot, velocity_dot  # noqa: RET504


# %%
# .. note::
#
#    The first parameter of an RHS function must be the time,
#    and the others must be the state of the system at this time.
#
# We want to solve the oscillator problem for a set of time values:

times = linspace(0.0, 10, 200)

# %%
# Step 2: Create a discipline
# ...........................
# Next, we create an :class:`.MDODiscipline` that will be used to build the
# :class:`.ODEDiscipline`:

rhs_discipline = create_discipline(
    "AutoPyDiscipline",
    py_func=compute_rhs_function,
    grammar_type=MDODiscipline.GrammarType.SIMPLE,
)

# %%
# Step 3: Create and solve the ODEDiscipline
# ..........................................
# The ``state_names`` are the names of the state parameters
# used as input for the ``compute_rhs_function``.
# These strings are used to create the grammar of the :class:`.ODEDiscipline`.
state_names = ["position", "velocity"]
state_solution_names = ["position_sol", "velocity_sol"]

ode_discipline = ODEDiscipline(
    discipline=rhs_discipline,
    times=times,
    state_names=state_names,
    return_trajectories=True,
    state_trajectory_names=state_solution_names,
)

local_data = ode_discipline.execute()

# %%
# Step 4: Visualize the result
# ............................
for state_variable_name in ("position_sol", "velocity_sol"):
    plt.plot(times, local_data[state_variable_name], label=state_variable_name)

plt.show()

# %%
# Shortcut
# --------
# The oscillator discipline is provided by |g| for direct use.
ode_discipline = OscillatorDiscipline(omega=4, times=times, return_trajectories=True)
ode_discipline.execute({
    "position": position_init,
    "velocity": velocity_init,
})
