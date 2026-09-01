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
"""# Tutorial - Solving an ODE with ODEProblem

## Goal

This tutorial introduces the
[ODEProblem][gemseo.ode.problem.ODEProblem]
and the
[ODESolverLibraryFactory][gemseo.ode.factory.ODESolverLibraryFactory],
the GEMSEO components for defining and solving a first-order ordinary
differential equation (ODE) of the form

$$\\frac{ds(t)}{dt} = f(t, s(t))$$

where $f$ is the right-hand side (RHS) function,
$t$ is time and $s(t)$ is the state vector.
Solving it requires initial conditions $s(t_0) = s_0$.

You will learn how to:

- **define** an ODE problem with
  [ODEProblem][gemseo.ode.problem.ODEProblem],
- **provide** an optional explicit Jacobian for improved performance,
- **solve** the problem with
  [ODESolverLibraryFactory][gemseo.ode.factory.ODESolverLibraryFactory],
- **inspect** the results.

As a running example, this tutorial uses the Van der Pol equation [@hartman2002],
describing the position over time of an oscillator with non-linear damping:

$$\\frac{d^2 x(t)}{dt^2} - \\mu (1 - x^2(t)) \\frac{dx(t)}{dt} + x(t) = 0$$

Rewritten as a first-order system by introducing $y = \\frac{dx}{dt}$
and the state vector $s = \\begin{pmatrix}x\\\\y\\end{pmatrix}$:

$$\\frac{ds(t)}{dt} = f(t, s(t))
= \\begin{pmatrix} y(t) \\\\ \\mu (1 - x^2(t)) y(t) - x(t) \\end{pmatrix}$$
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from numpy import array
from numpy import zeros

from gemseo.ode import RK45_Settings
from gemseo.ode.factory import ODESolverLibraryFactory
from gemseo.ode.problem import ODEProblem

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray

# %%
# ## Step 1 — Define the RHS function
#
# You define $f(t, s(t))$ as a Python function.
# The stiffness parameter $\mu$ controls the degree of non-linearity:
mu = 5


def evaluate_f(time: float, state: RealArray):
    """Evaluate the right-hand side function $f$.

    Args:
        time: The time at which $f$ should be evaluated.
        state: The state for which $f$ should be evaluated.

    Returns:
        The value of $f$ at `time` and `state`.
    """
    return array([state[1], mu * state[1] * (1 - state[0] ** 2) - state[0]])


# %%
# ## Step 2 — Create the ODEProblem
#
# An [ODEProblem][gemseo.ode.problem.ODEProblem] groups the RHS function,
# the initial state and the time interval into a single object:
initial_state = array([2, -2 / 3])
initial_time = 0.0
final_time = 50.0

ode_problem = ODEProblem(
    func=evaluate_f,
    initial_state=initial_state,
    times=(initial_time, final_time),
    solve_at_algorithm_times=True,
)


# %%
# ## Step 3 — Optionally provide an explicit Jacobian
#
# By default, the Jacobian of $f$ with respect to the state is approximated
# using finite differences.
# Providing an explicit Jacobian can improve accuracy and performance:
def evaluate_jac(time: float, state: RealArray):
    """Evaluate the Jacobian of $f$ with respect to the state.

    Args:
        time: The time at which the Jacobian should be evaluated.
        state: The state for which the Jacobian should be evaluated.

    Returns:
        The Jacobian of $f$ at `time` and `state`.
    """
    jac = zeros((2, 2))
    jac[1, 0] = -mu * 2 * state[1] * state[0] - 1
    jac[0, 1] = 1
    jac[1, 1] = mu * (1 - state[0] * state[0])
    return jac


ode_problem_with_jacobian = ODEProblem(
    evaluate_f,
    initial_state,
    (initial_time, final_time),
    jac_function_wrt_state=evaluate_jac,
)

# %%
# ## Step 4 — Solve the ODE problem
#
# Use [ODESolverLibraryFactory][gemseo.ode.factory.ODESolverLibraryFactory]
# to solve the problem.
# Here you use the Runge-Kutta RK45 method:
ODESolverLibraryFactory().execute(ode_problem, RK45_Settings())
ODESolverLibraryFactory().execute(ode_problem_with_jacobian, RK45_Settings())

# %%
# ## Step 5 — Inspect the results
#
# The solution is accessible via the `result` attribute of the
# [ODEProblem][gemseo.ode.problem.ODEProblem].
# Use
# [ODEResult.algorithm_has_converged][gemseo.ode.result.ODEResult.algorithm_has_converged]
# and
# [ODEResult.algorithm_termination_message][gemseo.ode.result.ODEResult.algorithm_termination_message]
# to check convergence,
# and
# [ODEResult.times][gemseo.ode.result.ODEResult.times]
# and
# [ODEResult.state_trajectories][gemseo.ode.result.ODEResult.state_trajectories]
# to access the solution:
plt.plot(ode_problem.result.times, ode_problem.result.state_trajectories[0], label="x")
plt.plot(ode_problem.result.times, ode_problem.result.state_trajectories[1], label="y")
plt.legend()
plt.xlabel("time")
plt.show()

# %%
# ## Key takeaways
#
# - [ODEProblem][gemseo.ode.problem.ODEProblem] groups the RHS function,
#   initial state and time interval into a single object.
# - By default, the Jacobian is approximated by finite differences.
#   Passing an explicit `jac_function_wrt_state` can improve performance.
# - [ODESolverLibraryFactory][gemseo.ode.factory.ODESolverLibraryFactory]
#   solves the problem; other algorithms can be selected beyond the default RK45.
# - The solution is available via `ode_problem.result.times` and
#   `ode_problem.result.state_trajectories`.
#
# !!! note
#
#     The [VanDerPol][gemseo.problem.ode.van_der_pol.VanDerPol] class provides
#     a ready-to-use implementation of the Van der Pol problem.
#     It can be imported from
#     [gemseo.problem.ode.van_der_pol][gemseo.problem.ode.van_der_pol]
#     and used directly without defining the RHS function manually.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Add a termination condition to an ODE][add-a-termination-condition-to-an-ode].
