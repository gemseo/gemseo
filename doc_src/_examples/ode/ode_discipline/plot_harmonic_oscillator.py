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
# Giulio Gargantini
"""
Solve an ODE: the harmonic oscillator
=====================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import array
from numpy import cos
from numpy import linspace
from numpy import ndarray
from numpy import pi
from numpy import sin

from gemseo import create_discipline
from gemseo.core.discipline.discipline import Discipline
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.problems.ode.oscillator_discipline import OscillatorDiscipline

# %%
# This tutorial describes how to instantiate an ODEDiscipline to solve a
# first-order ordinary differential equation.
# A first-order ODE is a differential equation that can be written as
#
# .. math::
#
#     \frac{ds(t)}{dt} = f(t, s(t))
#
# where the right-hand side of the equation :math:`f(t, s(t))` is a function of
# time :math:`t` and of a state :math:`s(t)` that returns another state
# :math:`\frac{ds(t)}{dt}` :cite:`hartman2002`. To solve this
# equation, initial conditions must be set:
#
# .. math::
#
#     s(t_0) = s_0
#
# For this example, we consider the harmonic oscillator equation as an example of a
# time-independent problem.
#
# Solving the harmonic oscillator problem
# ---------------------------------------
#
# The harmonic problem describes the position over time of a body oscillating with a
# frequency :math:`\omega/ (2 \pi)` according to the following ODE:
#
# .. math::
#
#     \frac{d^2 x(t)}{dt^2} + \omega^2 x(t) = 0
#
# where :math:`x(t)` is the position coordinate at time :math:`t`.
#
# To solve the harmonic oscillator problem with |g|,
# we start by modeling this second-order equation as a first-order equation.
# Let us define :math:`y = \frac{dx}{dt}` and the state vector :math:`s` as
# :math:`s = \begin{pmatrix}x\\y\end{pmatrix}`.
# Then the harmonic oscillator problem can be rewritten as:
#
# .. math::
#
#     \frac{ds(t)}{dt} = f(t, s(t))
#     = \begin{pmatrix} y(t) \\ - \omega^2 x(t) \end{pmatrix}
#
#
#
# The analytical solution of the harmonic oscillator problem characterized by an angular
# velocity :math:`\omega`, and with initial position and velocity :math:`x_0` and
# :math:`v_0` respectively, at the initial time :math:`t = 0` is:
#
# .. math::
#
#     x(t) = x_0 \cos(\omega t) + (v_0 / \omega) \sin(\omega t)

# %%
# Step 1 : Definition of the dynamic
# ..................................
# As the first step, we introduce a discipline defining the dynamics of the problem,
# that is a discipline representing the right-hand side of the ODE.
# For a harmonic oscillator, the discipline can be an :class:`.AutoPyDiscipline`.
#
# The discipline describing the RHS must include, among its inputs, the time variable
# and the variables defining the state (in the case of the harmonic oscillator, the
# variables ``position`` and ``velocity``). The outputs of the discipline must be the
# time derivatives of the state variables (here ``position_dot`` and ``velocity_dot``).
# All inputs that are neither the time variable nor the state variables, are denoted as
# design variables.

_time = array([0.0])
initial_position_1 = array([1.0])
initial_velocity_1 = array([0.0])
omega_1 = array([2.0])


def rhs_function(
    time: ndarray = _time,
    position: ndarray = initial_position_1,
    velocity: ndarray = initial_velocity_1,
    omega: ndarray = omega_1,
):
    position_dot = velocity
    velocity_dot = -(omega**2) * position

    return position_dot, velocity_dot  # noqa: RET504


rhs_discipline = create_discipline(
    "AutoPyDiscipline",
    py_func=rhs_function,
    grammar_type=Discipline.GrammarType.SIMPLE,
)

# %%
# Step 2: Initialization of the ODEDiscipline
# ...........................................
#
# Once the discipline representing the right-hand side of the ODE has been created,
# we can create an instance of :class:`.ODEDiscipline`, representing the initial-value
# problem to be solved.
#
# The constructor of the :class:`.ODEDiscipline` must be provided with the arguments
# ``discipline`` (the discipline representing the RHS of the ODE), and ``times`` (an
# :type:`ArrayLike` with at least two entries: the fist representing the initial time,
# and the last one the final time).
# The parameter ``state_names`` is a list of the name of the state variables in
# ``rhs_discipline``, in order  to differentiate them from the time variable (named
# ``"time"`` by default), and from the design variables (here ``omega``).
# By default, the output of the ``ODEDiscipline`` are: the final time of the evaluation
# of the ODE (``"termination_time"``), the list of times for which the ODE has been
# solved (``"times"``), and the final states of the state  variables (named by default
# ``"X_final"``, where ``"X"`` is the name of the corresponding  state variable).
# If the boolean ``return_trajectories`` is set to ``True``, additional outputs is
# provided, consisting in the trajectories of the state variables in the times listed
# in the output ``"times"``.

ode_discipline = ODEDiscipline(
    rhs_discipline=rhs_discipline,
    times=linspace(0.0, 10.0, 51),
    state_names=["position", "velocity"],
    return_trajectories=True,
)

# %%
# Step 3: Execution of the ODEDiscipline
# ......................................
#
# Once the :class:`.ODEDiscipline` has been initialized, it can be executed like all
# other disciplines in |g| by the method :meth:`execute`.

ode_res_1 = ode_discipline.execute()

# %%
# The default inputs of the :class:`.ODEDiscipline` are the default inputs of the
# underlying discipline defining the RHS of the ODE. Therefore, ``ode_res_1`` contains
# the solution of the problem of a harmonic oscillator with angular velocity `omega`
# equal to :math:`2.0`, with initial position :math:`1.0` and initial velocity
# :math:`0.0`.
#
# Different values for the design variables and for the initial conditions can be
# specified by passing a suitable dictionary to the `execute()` method of the
# :class:`.ODEDiscipline`.

initial_position_2 = array([2.0])
initial_velocity_2 = array([0.5])
omega_2 = array([1.0])

ode_res_2 = ode_discipline.execute({
    "initial_position": initial_position_2,
    "initial_velocity": initial_velocity_2,
    "omega": omega_2,
})

# %%
# The object ``ode_res_2`` contains the solution of a different harmonic oscillator
# problem, characterized by an angular velocity ``omega`` equal to :math:`1.0`, with
# a positive initial velocity of :math:`0.5`, and an initial position deviated by
# :math:`2.0` from the equilibrium.

# %%
# Terminating events
# ..................
#
# In some cases, it can be useful not to pursue the solution of the ODE for the entire
# time interval, but only up to the realization of a certain condition. For example, one
# could be interested in the dynamic of a falling object up to its impact on the ground.
# In solvers like :mod:`scipy.integrate.solve_ivp`, the termination condition is encoded by
# a function with the same entries as the RHS of the ODE, returning a real value.
# The termination condition is fulfilled when the function crosses the threshold
# :math:`0` (for further information, consult `the SciPy documentation
# <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_).
# In |g|, the same result can be achieved by passing a list of disciplines to the
# constructor of :class:`.ODEDiscipline`, each representing a termination condition.
# All disciplines representing termination conditions must have the same inputs as the
# discipline for the dynamic, and one real output.
#
# In this example, we consider a new :class:`.ODEDiscipline`, describing the trajectory
# of a harmonic oscillator from its starting point up to the instant when it crosses
# the equilibrium position.

initial_position_3 = array([1.5])


def termination_function(
    time: ndarray = _time,
    position: ndarray = initial_position_3,
    velocity: ndarray = initial_velocity_1,
    omega: ndarray = omega_1,
):
    termination = position
    return termination  # noqa: RET504


termination_discipline = create_discipline(
    "AutoPyDiscipline",
    py_func=termination_function,
    grammar_type=Discipline.GrammarType.SIMPLE,
)

# %%
# Here, ``termination_discipline`` is the :class:`.AutoPyDiscipline` encoding the
# termination condition. In order to include this condition in the solution of the ODE,
# we pass the tuple ``(termination_discipline,)`` as the argument
# ``termination_event_disciplines`` of the constructor of :class:`.ODEDiscipline`.

ode_discipline_termination = ODEDiscipline(
    rhs_discipline=rhs_discipline,
    times=linspace(0.0, 10.0, 51),
    state_names=["position", "velocity"],
    termination_event_disciplines=(termination_discipline,),
    return_trajectories=True,
    solve_at_algorithm_times=True,
)

ode_res_termination = ode_discipline_termination.execute({
    "initial_position": initial_position_3,
    "initial_velocity": initial_velocity_1,
    "omega": omega_1,
})

# %%
# Examining the results
# .....................
#
# The outcomes of the discipline executions can be accessed by passing the names of
# the corresponding outputs to ``ode_res_1``, ``ode_res_2``, and ``ode_res_termination``.

plt.plot(ode_res_1["times"], ode_res_1["position"])
plt.plot(ode_res_2["times"], ode_res_2["position"])
plt.plot(ode_res_termination["times"], ode_res_termination["position"])
plt.legend(["ode_res_1", "ode_res_2", "ode_res_termination"])
plt.title("Harmonic oscillators with different frequencies")
plt.show()


# %%
# These results can also be compared with the analytical solutions.

analytic_res_1 = initial_position_1 * cos(omega_1 * ode_res_1["times"]) + (
    initial_velocity_1 / omega_1
) * sin(omega_1 * ode_res_1["times"])
analytic_res_2 = initial_position_2 * cos(omega_2 * ode_res_2["times"]) + (
    initial_velocity_2 / omega_2
) * sin(omega_2 * ode_res_2["times"])
analytic_res_termination = initial_position_3 * cos(
    omega_1 * ode_res_termination["times"]
) + initial_velocity_1 / omega_1 * sin(omega_1 * ode_res_termination["times"])

plt.plot(ode_res_1["times"], analytic_res_1, "r", label="Analytical solution")
plt.plot(
    ode_res_1["times"],
    ode_res_1["position"],
    "b--",
    label="Numerical solution",
)
plt.legend(["Analytical solution", "Solution by ODEDiscipline"])
frequency = omega_1[0] / (2 * pi)
title = f"Harmonic oscillator with frequency {omega_1[0]}/($2 \\pi$) = {frequency:.3f}"
plt.title(title)
plt.show()

plt.plot(ode_res_2["times"], analytic_res_2, "r", label="Analytical solution")
plt.plot(
    ode_res_2["times"],
    ode_res_2["position"],
    "b--",
    label="Numerical solution",
)
plt.legend(["Analytical solution", "Solution by ODEDiscipline"])
frequency = omega_2[0] / (2 * pi)
title = f"Harmonic oscillator with frequency {omega_2[0]}/($2 \\pi$) = {frequency:.3f}"
plt.title(title)
plt.show()


plt.plot(
    ode_res_termination["times"],
    analytic_res_termination,
    "r",
    label="Analytical solution",
)
plt.plot(
    ode_res_termination["times"],
    ode_res_termination["position"],
    "b--",
    label="Numerical solution",
)
plt.plot(
    ode_res_1["times"],
    [0.0] * len(ode_res_1["times"]),
    "k--",
    label="termination threshold = 0.0",
)
plt.legend([
    "Analytical solution",
    "Solution by ODEDiscipline",
    "termination threshold = 0.0",
])
title = "Harmonic oscillator with terminating condition"
plt.title(title)
plt.show()


# %%
# Shortcut
# ........
#
# The class :class:`.OscillatorDiscipline` is available in the package
# :mod:`gemseo.problems.ode`, so it just needs to be imported to be used.
ode_oscillator_discipline = OscillatorDiscipline(
    omega=1.0, times=linspace(0.0, 10.0, 51)
)
ode_oscillator_discipline.execute()
