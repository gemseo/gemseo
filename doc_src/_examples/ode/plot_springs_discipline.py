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
Solve a system of coupled ODEs
==============================
"""

from __future__ import annotations

from itertools import starmap

from matplotlib import pyplot as plt
from numpy import linspace
from numpy.random import default_rng

from gemseo import create_discipline
from gemseo.core.chains.chain import MDOChain
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.problems.ode._springs import Mass
from gemseo.problems.ode._springs import create_chained_masses

# %%
# This tutorial describes how to use the :class:`.ODEDiscipline` with coupled ODEs.
#
# Problem description
# -------------------
#
# Consider a set of point masses with masses :math:`m_1,\ m_2,...\ m_n` connected by
# springs with stiffnesses :math:`k_1,\ k_2,...\ k_{n+1}`. The springs at each end of the
# system are connected to fixed points. We hereby study the response of the system to the
# displacement of one of the point masses.
#
# .. figure:: ../_images/ode/springs.png
#   :width: 400
#   :alt: Illustration of the springs-masses problem.
#
# The motion of each point mass in this system is described
# by the following set of ordinary differential equations (ODEs):
#
# .. math::
#
#    \left\{ \begin{cases}
#        \frac{dx_i}{dt} &= v_i \\
#        \frac{dv_i}{dt} &=
#            - \frac{k_i + k_{i+1}}{m_i}x_i
#            + \frac{k_i}{m_i}x_{i-1} + \frac{k_{i+1}}{m_i}x_{i+1}
#    \end{cases} \right.
#
# where :math:`x_i` is the position of the :math:`i`-th point mass
# and :math:`v_i` is its velocity.
#
# These equations are coupled, since the forces applied to any given mass depend on the
# positions of its neighbors. In this tutorial, we will use the framework of the
# :class:`.ODEDisciplines` to solve this set of coupled equations.
#
# Using an :class:`.ODEDiscipline` to solve the problem
# -----------------------------------------------------
#
# Let's consider the problem described above in the case of two masses. First we describe
# the right-hand side (RHS) function of the equations of motion for each point mass.

stiffness_0 = 1
stiffness_1 = 1
stiffness_2 = 1
mass_0 = 1
mass_1 = 1
initial_position_0 = 1
initial_position_1 = 0
initial_velocity_0 = 0
initial_velocity_1 = 0

# Vector of times at which to solve the problem.
times = linspace(0, 1, 30)


def compute_mass_0_rhs(
    time=0,
    position_0=initial_position_0,
    velocity_0=initial_velocity_0,
    position_1=initial_position_1,
):
    """Compute the RHS of the ODE associated with the first point mass.

    Args:
        time: The time at which to evaluate the RHS.
        position_0: The position of the first point mass at this time.
        velocity_0: The velocity of the first point mass at this time.
        position_1: The position of the second point mass at this time.

    Returns:
        The first- and second-order derivatives of the position
        of the first point mass.
    """
    position_0_dot = velocity_0
    velocity_0_dot = (
        -(stiffness_0 + stiffness_1) * position_0 + stiffness_1 * position_1
    ) / mass_0
    return position_0_dot, velocity_0_dot


def compute_mass_1_rhs(
    time=0,
    position_1=initial_position_1,
    velocity_1=initial_velocity_1,
    position_0=initial_position_0,
):
    """Compute the RHS of the ODE associated with the secondpoint mass.

    Args:
        time: The time at which to evaluate the RHS.
        position_1: The position of the second point mass at this time.
        velocity_1: The velocity of the second point mass at this time.
        position_0: The position of the first point mass at this time.

    Returns:
        The first- and second-order derivatives of the position
        of the second point mass.
    """
    position_1_dot = velocity_1
    velocity_1_dot = (
        -(stiffness_1 + stiffness_2) * position_1 + stiffness_1 * position_0
    ) / mass_1
    return position_1_dot, velocity_1_dot


# %%
# We can then create a list of :class:`.ODEDiscipline` objects
#

rhs_disciplines = [
    create_discipline("AutoPyDiscipline", py_func=compute_rhs)
    for compute_rhs in [compute_mass_0_rhs, compute_mass_1_rhs]
]
ode_disciplines = [
    ODEDiscipline(
        rhs_discipline,
        times,
        state_names=[f"position_{i}", f"velocity_{i}"],
        return_trajectories=True,
        rtol=1e-12,
        atol=1e-12,
    )
    for i, rhs_discipline in enumerate(rhs_disciplines)
]
for ode_discipline in ode_disciplines:
    ode_discipline.execute()

# %%
# We apply an MDA with the Gauss-Seidel algorithm:
mda = MDAGaussSeidel(ode_disciplines)
local_data = mda.execute()

# %%
# We can plot the residuals of this MDA.

mda.plot_residual_history()


# %%
# Plotting the solution
# ---------------------
plt.plot(times, local_data["position_0_trajectory"], label="mass 0")
plt.plot(times, local_data["position_1_trajectory"], label="mass 1")
plt.legend()
plt.show()

# %%
# Another formulation
# -------------------
#
# In the previous section, we considered the time-integration within each ODE discipline,
# then coupled the disciplines, as illustrated in the next figure.
#
# .. figure:: ../_images/ode/coupling.png
#   :width: 400
#   :alt: Integrate, then couple.
#
# Another possibility to tackle this problem is to define the couplings within a
# discipline, as illustrated in the next figure.
#
# .. figure:: ../_images/ode/time_integration.png
#   :width: 400
#   :alt: Couple, then integrate.
#
# To do so, we can use the RHS disciplines we created earlier to define an
# :class:`.MDOChain`.

mda = MDOChain(rhs_disciplines)

# %%
# We then define the ODE discipline that contains the couplings and execute it.

ode_discipline = ODEDiscipline(
    mda,
    times,
    state_names=["position_0", "velocity_0", "position_1", "velocity_1"],
    return_trajectories=True,
    rtol=1e-12,
    atol=1e-12,
)
local_data = ode_discipline.execute()

plt.plot(times, local_data["position_0_trajectory"], label="mass 0")
plt.plot(times, local_data["position_1_trajectory"], label="mass 1")
plt.legend()
plt.show()


# %%
# Shortcut
# --------
# The :mod:`.springs` module provides a shortcut to access this problem. The user can define
# a list of masses, stiffnesses and initial positions, then create all the disciplines
# with a single call.
rng = default_rng(123)
masses = rng.random(3)
stiffnesses = rng.random(4)
positions = [1, 0, 0]
masses = list(starmap(Mass, zip(masses, stiffnesses[:-1], positions)))
chained_masses = create_chained_masses(stiffnesses[-1], *masses)
mda = MDOChain(chained_masses)
mda.execute()
