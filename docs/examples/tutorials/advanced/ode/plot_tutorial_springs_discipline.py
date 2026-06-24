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
"""# Tutorial - Solving a system of coupled ODEs

## Goal

This tutorial shows how to solve a system of coupled ordinary differential equations
(ODEs) with GEMSEO, using the coupled springs problem as a running example.

When ODEs are coupled - meaning the dynamics of each state variable depend on
the values of other state variables - two strategies are available:

- **Strategy 1 - MDA between ODEDiscipline instances**:
  create one [ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline]
  per subsystem and couple them via a
  [MDAGaussSeidel][gemseo.mda.gauss_seidel.MDAGaussSeidel].
  Each discipline integrates its own ODE in time;
  the MDA iterates until the coupling variables converge.

- **Strategy 2 - Single ODEDiscipline with coupled dynamics**:
  define all the coupled dynamics inside a single RHS discipline
  (here an [DisciplineChain][gemseo.core.chains.chain.DisciplineChain])
  and wrap it in a single
  [ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline].
  The coupling is resolved at each time step during the integration.

Both strategies are presented and compared on the same problem.

## The coupled springs problem

Consider $n$ point masses $m_1, m_2, \\ldots, m_n$ connected in a chain by $n+1$ springs
with stiffnesses $k_1, k_2, \\ldots, k_{n+1}$.
The leftmost and rightmost springs are attached to fixed walls.

![image](../../../../../assets/images/ode/springs.png)

*The figure above illustrates the case $n = 3$.*

The motion of each mass $m_i$ ($i = 1, \\ldots, n$) is described by:

$$
\\begin{cases}
    \\frac{dx_i}{dt} &= v_i \\\\
    \\frac{dv_i}{dt} &=
        -\\frac{k_i + k_{i+1}}{m_i} x_i
        + \\frac{k_i}{m_i} x_{i-1}
        + \\frac{k_{i+1}}{m_i} x_{i+1}
\\end{cases}
$$

where $x_i$ and $v_i$ are the position and velocity of mass $i$,
and the boundary conditions $x_0 = x_{n+1} = 0$ encode the fixed walls.
The equations are coupled: the force on each mass depends on its neighbours' positions.

This tutorial focuses on $n = 2$ masses, for which the equations become:

$$
\\begin{cases}
    \\frac{dx_1}{dt} &= v_1 \\\\
    \\frac{dv_1}{dt} &= -\\frac{k_1 + k_2}{m_1} x_1 + \\frac{k_2}{m_1} x_2 \\\\
    \\frac{dx_2}{dt} &= v_2 \\\\
    \\frac{dv_2}{dt} &= \\frac{k_2}{m_2} x_1 - \\frac{k_2 + k_3}{m_2} x_2
\\end{cases}
$$
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from numpy import array
from numpy import linspace
from scipy.interpolate import interp1d

from gemseo.algos.ode.scipy_ode.settings.rk45 import RK45_Settings
from gemseo.core.chains.chain import DisciplineChain
from gemseo.core.discipline import Discipline
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

# %%
# ## Step 1 - Define the problem parameters
#
stiffness_1 = 0.6
stiffness_2 = 1
stiffness_3 = 2
mass_1 = 1
mass_2 = 1

initial_position_1 = 1
initial_position_2 = 0
initial_velocity_1 = 0
initial_velocity_2 = 0

times = linspace(0.0, 2.0, 30)


# %%
# ## Step 2 - Strategy 1: MDA between ODEDiscipline instances
#
# You define one RHS discipline per mass.
# Each discipline takes the trajectory of the neighbouring mass
# as an input (interpolated at each time step)
# to account for the coupling:
class RHSMassDisciplineLeft(Discipline):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.io.input_grammar.update_from_names((
            "time",
            "position_1",
            "velocity_1",
            "position_2",
        ))
        self.io.output_grammar.update_from_names(("position_1_dot", "velocity_1_dot"))
        self.default_input_data = {
            "time": 0.0,
            "position_1": array([initial_position_1]),
            "velocity_1": array([initial_velocity_1]),
            "position_2": times * 0.0,
        }
        self.add_differentiated_inputs(["position_1", "velocity_1"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping:
        time = input_data["time"]
        position_1 = input_data["position_1"]
        velocity_1 = input_data["velocity_1"]
        position_2_vec = input_data["position_2"]
        position_2 = interp1d(times, position_2_vec, assume_sorted=True)(time)
        return {
            "position_1_dot": velocity_1,
            "velocity_1_dot": (
                -(stiffness_1 + stiffness_2) * position_1 + stiffness_2 * position_2
            )
            / mass_1,
        }


class RHSMassDisciplineRight(Discipline):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.io.input_grammar.update_from_names((
            "time",
            "position_2",
            "velocity_2",
            "position_1",
        ))
        self.io.output_grammar.update_from_names(("position_2_dot", "velocity_2_dot"))
        self.default_input_data = {
            "time": 0.0,
            "position_2": array([initial_position_2]),
            "velocity_2": array([initial_velocity_2]),
            "position_1": times * 0.0,
        }
        self.add_differentiated_inputs(["position_2", "velocity_2"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping:
        time = input_data["time"]
        position_2 = input_data["position_2"]
        velocity_2 = input_data["velocity_2"]
        position_1_vec = input_data["position_1"]
        position_1 = interp1d(times, position_1_vec, assume_sorted=True)(time)
        return {
            "position_2_dot": velocity_2,
            "velocity_2_dot": (
                -(stiffness_2 + stiffness_3) * position_2 + stiffness_2 * position_1
            )
            / mass_2,
        }


# %%
# You create one [ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline]
# per mass and couple them with a
# [MDAGaussSeidel][gemseo.mda.gauss_seidel.MDAGaussSeidel].
# ![image](../../../../../assets/images/ode/springs-disciplines.png)
#
# *The figure above illustrates the case $n = 3$.*
rhs_disciplines = [RHSMassDisciplineLeft(), RHSMassDisciplineRight()]

ode_disciplines = [
    ODEDiscipline(
        rhs_discipline=rhs_discipline,
        times=times,
        state_names=(f"position_{i}", f"velocity_{i}"),
        time_name="time",
        return_trajectories=True,
        ode_solver_settings=RK45_Settings(
            rtol=1e-6,
            atol=1e-6,
        ),
    )
    for i, rhs_discipline in enumerate(rhs_disciplines, start=1)
]
for ode_discipline in ode_disciplines:
    ode_discipline.execute()

mda = MDAGaussSeidel(ode_disciplines)
result_strategy_1 = mda.execute()

plt.plot(times, result_strategy_1["position_1"], label="Mass 1")
plt.plot(times, result_strategy_1["position_2"], label="Mass 2")
plt.title("Strategy 1 - MDA between ODEDiscipline instances")
plt.legend()
plt.show()


# %%
# ## Step 3 - Strategy 2: single ODEDiscipline with coupled dynamics
#
# You define the RHS of both masses as plain Python functions
# and wrap them in an [DisciplineChain][gemseo.core.chains.chain.DisciplineChain].
# The coupling is then resolved internally at each time step
# during the integration.
# ![image](../../../../../assets/images/ode/time_integration.png)


def compute_mass_1_rhs(
    time=0,
    position_1=initial_position_1,
    velocity_1=initial_velocity_1,
    position_2=initial_position_2,
):
    position_1_dot = velocity_1
    velocity_1_dot = (
        -(stiffness_1 + stiffness_2) * position_1 + stiffness_2 * position_2
    ) / mass_1
    return position_1_dot, velocity_1_dot


def compute_mass_2_rhs(
    time=0,
    position_2=initial_position_2,
    velocity_2=initial_velocity_2,
    position_1=initial_position_1,
):
    position_2_dot = velocity_2
    velocity_2_dot = (
        -(stiffness_2 + stiffness_3) * position_2 + stiffness_2 * position_1
    ) / mass_2
    return position_2_dot, velocity_2_dot


rhs_disciplines = [
    AutoPyDiscipline(py_func=compute_rhs)
    for compute_rhs in [compute_mass_1_rhs, compute_mass_2_rhs]
]
rhs_disciplines[0].add_differentiated_inputs(["time", "position_1", "velocity_1"])
rhs_disciplines[1].add_differentiated_inputs(["time", "position_2", "velocity_2"])

mda_chain = DisciplineChain(rhs_disciplines)

ode_discipline = ODEDiscipline(
    rhs_discipline=mda_chain,
    state_names={
        "position_1": "position_1_dot",
        "velocity_1": "velocity_1_dot",
        "position_2": "position_2_dot",
        "velocity_2": "velocity_2_dot",
    },
    return_trajectories=True,
    times=times,
    ode_solver_settings=RK45_Settings(
        rtol=1e-12,
        atol=1e-12,
    ),
)
result_strategy_2 = ode_discipline.execute()

plt.plot(times, result_strategy_2["position_1"], label="Mass 1")
plt.plot(times, result_strategy_2["position_2"], label="Mass 2")
plt.title("Strategy 2 - Single ODEDiscipline with coupled dynamics")
plt.legend()
plt.show()

# %%
# ## Step 4 - Compare the two strategies
#
# Both strategies should yield the same trajectories.
# You verify this by plotting the absolute difference between the results:
error_1 = abs(result_strategy_1["position_1"] - result_strategy_2["position_1"])
error_2 = abs(result_strategy_1["position_2"] - result_strategy_2["position_2"])

plt.plot(times, error_1, label="Mass 1")
plt.plot(times, error_2, label="Mass 2")
plt.title("Absolute difference between the two strategies")
plt.legend()
plt.show()

# %%
# ## Key takeaways
#
# - When ODEs are coupled, two strategies are available in GEMSEO:
#   couple multiple [ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline]
#   instances via an MDA,
#   or define all coupled dynamics inside a single RHS discipline
#   wrapped in one [ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline].
# - **Strategy 1** (MDA) integrates each subsystem independently and iterates
#   on the coupling variables across disciplines.
#   It is more modular but requires the full trajectory of coupling variables
#   to be passed between disciplines.
# - **Strategy 2** (single ODEDiscipline) resolves the coupling at each time step
#   during the integration, using an
#   [DisciplineChain][gemseo.core.chains.chain.DisciplineChain] as the RHS.
#   It is more compact and avoids trajectory interpolation.
# - Both strategies produce equivalent results for the same problem.
#
# !!! note
#
#     The [CoupledSpringsGenerator][gemseo.problems.ode.springs.coupled_springs_generator.CoupledSpringsGenerator]
#     class provides a ready-to-use implementation of the coupled springs problem
#     for any number of masses.
#     It can generate both coupled
#     [ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline]
#     instances and a single discipline with coupled dynamics,
#     via `create_coupled_ode_disciplines()` and
#     `create_discipline_with_coupled_dynamics()` respectively.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Add a termination condition to an ODE][add-a-termination-condition-to-an-ode]
