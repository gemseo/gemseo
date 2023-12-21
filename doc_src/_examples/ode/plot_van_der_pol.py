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
Solve an ODE: the Van der Pol problem
=====================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import array
from numpy import zeros

from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.algos.ode.ode_solvers_factory import ODESolversFactory
from gemseo.problems.ode.van_der_pol import VanDerPol

# %%
# This tutorial describes how to solve an ordinary differential equation (ODE)
# problem with |g|. A first-order ODE is a differential equation that can be
# written as
#
# .. math::
#
#     \frac{ds(t)}{dt} = f(t, s(t))
#
# where the right-hand side of the equation :math:`f(t, s(t))` is a function of
# time :math:`t` and of a state :math:`s(t)` that returns another state
# :math:`\frac{ds(t)}{dt}` (see Hartman, Philip (2002) [1964],
# Ordinary differential equations, Classics in Applied Mathematics, vol. 38,
# Philadelphia: Society for Industrial and Applied Mathematics). To solve this
# equation, initial conditions must be set:
#
# .. math::
#
#     s(t_0) = s_0
#
# For this example, we consider the Van der Pol equation as an example of a
# time-independent problem.
#
# Solving the Van der Pol time-independent problem
# ------------------------------------------------
#
# The Van der Pol problem describes the position over time of an oscillator with
# non-linear damping:
#
# .. math::
#
#     \frac{d^2 x(t)}{dt^2} -
#     \mu (1-x^2(t)) \frac{dx(t)}{dt} + x(t) = 0
#
# where :math:`x(t)` is the position coordinate at time :math:`t`, and
# :math:`\mu` is the stiffness parameter.
#
# To solve the Van der Pol problem with |g|, we first need to model this
# second-order equation as a first-order equation. Let
# :math:`y = \frac{dx}{dt}` and
# :math:`s = \begin{pmatrix}x\\y\end{pmatrix}`. Then the Van der Pol problem can be
# rewritten:
#
# .. math::
#
#     \frac{ds(t)}{dt} = f(t, s(t))
#     = \begin{pmatrix} y(t) \\ \mu (1-x^2(t)) y(t) - x(t) \end{pmatrix}


# %%
# Step 1 : Defining the problem
# .............................

mu = 5


def evaluate_f(time, state):
    return state[1], mu * state[1] * (1 - state[0] ** 2) - state[0]


initial_state = array([2, -2 / 3])
initial_time = 0.0
final_time = 50.0
ode_problem = ODEProblem(evaluate_f, initial_state, initial_time, final_time)


# %%
# By default, the Jacobian of the problem is approximated using the finite
# difference method. However, it is possible to define an explicit expression
# of the Jacobian and pass it to the problem. In the case of the Van der Pol
# problem, this would be:


def evaluate_jac(time, state):
    jac = zeros((2, 2))
    jac[1, 0] = -mu * 2 * state[1] * state[0] - 1
    jac[0, 1] = 1
    jac[1, 1] = mu * (1 - state[0] * state[0])
    return jac


ode_problem = ODEProblem(
    evaluate_f, initial_state, initial_time, final_time, jac=evaluate_jac
)

# %%
# Step 2: Solving the ODE problem
# ...............................
#
# Whether the Jacobian is specified or not, once the problem is defined, the ODE
# solver is called on the :class:`.ODEProblem` by using the :class:`.ODESolversFactory`:
ODESolversFactory().execute(ode_problem)

# %%
# By default, the Runge-Kutta method of order 4(5) (``"RK45"``) is used, but other
# algorithms can be applied by specifying the option ``algo_name`` in
# :meth:`ODESolversFactory().execute`.
# See more information on available algorithms in
# `the SciPy documentation
# <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_.
#
#
# Step 3: Examining the results
# .............................
#
# The convergence of the algorithm can be known by examining
# :attr:`.ODEProblem.is_converged` and :attr:`.ODEProblem.solver_message`.
#
# The solution of the :class:`.ODEProblem` on the user-specified time interval
# can be accessed through the vectors :attr:`.ODEProblem.states` and
# :attr:`.ODEProblem.times`.

plt.plot(ode_problem.result.time_vector, ode_problem.result.state_vector[0])
plt.plot(ode_problem.result.time_vector, ode_problem.result.state_vector[1])
plt.show()

# %%
# Shortcut
# ........
#
# The class :class:`.VanDerPol` is available in the package
# :mod:`gemseo.problems.ode`, so it just needs to be imported to be used.
ode_problem = VanDerPol()
ODESolversFactory().execute(ode_problem)
