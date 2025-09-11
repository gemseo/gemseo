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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: François Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
How to solve an optimization problem
====================================
"""

# %%
# Although the |g| library is dedicated to the :term:`MDO`, it can also be used
# for mono-disciplinary optimization problems.
# This example presents some analytical test cases.

# %%
# Imports
# *******
from __future__ import annotations

import numpy as np
from scipy import optimize

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import execute_post
from gemseo import get_available_opt_algorithms

# %%
# Optimization based on a design of experiments
# *********************************************
# Let :math:`(P)` be a simple optimization problem:
#
# .. math::
#
#    (P) = \left\{
#    \begin{aligned}
#      & \underset{x\in\mathbb{N}^2}{\text{minimize}}
#      & & f(x) = x_1 + x_2 \\
#      & \text{subject to}
#      & & -5 \leq x \leq 5
#    \end{aligned}
#    \right.
#
# In this section, we will see how to use |g| to solve this problem :math:`(P)` by
# means of a Design Of Experiments (:term:`DOE`)
#
# Define the objective function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Firstly, by means of the :func:`.create_discipline` high-level function,
# we create a :class:`.Discipline` of :class:`.AutoPyDiscipline` type
# from a Python function.


def f(x1=0.0, x2=0.0):
    y = x1 + x2
    return y


discipline = create_discipline("AutoPyDiscipline", py_func=f)

# %%
# Now, we want to minimize this :class:`.Discipline` over a design of experiments (DOE).

# %%
# Define the design space
# ~~~~~~~~~~~~~~~~~~~~~~~
# For that, by means of the :func:`.create_design_space` API function,
# we define the :class:`.DesignSpace` :math:`[-5, 5]\times[-5, 5]`
# by using its :meth:`~.DesignSpace.add_variable` method.

design_space = create_design_space()
design_space.add_variable("x1", 1, lower_bound=-5, upper_bound=5, type_="integer")
design_space.add_variable("x2", 1, lower_bound=-5, upper_bound=5, type_="integer")

# %%
# Define the DOE scenario
# ~~~~~~~~~~~~~~~~~~~~~~~
# Then, by means of the :func:`.create_scenario` high-level function,
# we define a :class:`.DOEScenario` from the :class:`.Discipline`
# and the :class:`.DesignSpace` defined above:

scenario = create_scenario(
    discipline,
    "y",
    design_space,
    formulation_name="DisciplinaryOpt",
    scenario_type="DOE",
)

# %%
# Execute the DOE scenario
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Lastly, we solve the :class:`.OptimizationProblem` included in the
# :class:`.DOEScenario` defined above by minimizing the objective function over a
# design of experiments included in the :class:`.DesignSpace`.
# Precisely, we choose a
# `full factorial design <https://en.wikipedia.org/wiki/Factorial_experiment>`_
# of size :math:`11^2`:

scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=11**2)

# %%
# The optimum results can be found in the execution log. It is also possible to
# extract them from the :attr:`.BaseScenario.optimization_result`.

optimization_result = scenario.optimization_result
print(
    f"The solution of P is (x*,f(x*)) = ({optimization_result.x_opt}, {optimization_result.f_opt})"
)

# %%
# Optimization based on a quasi-Newton method by means of the `SciPy <https://scipy.org/>`_ library
# *************************************************************************************************
# Let :math:`(P)` be a simple optimization problem:
#
# .. math::
#
#    (P) = \left\{
#    \begin{aligned}
#      & \underset{x}{\text{minimize}}
#      & & f(x) = \sin(x) - \exp(x) \\
#      & \text{subject to}
#      & & -2 \leq x \leq 2
#    \end{aligned}
#    \right.
#
# In this section, we will see how to use |g| to solve this problem :math:`(P)`
# by means of an optimizer directly used from the `SciPy <https://scipy.org/>`_ library.
#
# Define the objective function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Firstly, we create the objective function and its gradient as standard Python
# functions:


def g(x=0):
    y = np.sin(x) - np.exp(x)
    return y


def dgdx(x=0):
    y = np.cos(x) - np.exp(x)
    return y


# %%
# Minimize the objective function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now, we can minimize this Python function over its design space by means of
# the `L-BFGS-B algorithm <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_
# implemented in the function ``scipy.optimize.fmin_l_bfgs_b``.

x_0 = -0.5 * np.ones(1)
opt = optimize.fmin_l_bfgs_b(g, x_0, fprime=dgdx, bounds=[(-0.2, 2.0)])
x_opt, f_opt, _ = opt

# %%
# Then, we can display the solution of our optimization problem with the following code:

print(f"The solution of P is (x*,f(x*)) = ({x_opt[0]}, {f_opt}).")

# %%
# .. seealso::
#
#    You can find the SciPy implementation of the L-BFGS-B algorithm
#    `by clicking here
#    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_.

# %%
# Optimization based on a quasi-Newton method by means of the |g| optimization interface
# **************************************************************************************
# Let :math:`(P)` be a simple optimization problem:
#
# .. math::
#
#    (P) = \left\{
#    \begin{aligned}
#      & \underset{x}{\text{minimize}}
#      & & f(x) = \sin(x) - \exp(x) \\
#      & \text{subject to}
#      & & -2 \leq x \leq 2
#    \end{aligned}
#    \right.
#
# In this section, we will see how to use |g| to solve this problem :math:`(P)`
# by means of an optimizer
# from `SciPy <https://scipy.org/>`_ called through the optimization interface of |g|.
#
# Define the objective function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Firstly, by means of the :func:`.create_discipline` high-level function,
# we create an :class:`.Discipline` of :class:`.AutoPyDiscipline` type
# from a Python function:


def g(x=0):
    y = np.sin(x) - np.exp(x)
    return y


def dgdx(x=0):
    y = np.cos(x) - np.exp(x)
    return y


discipline = create_discipline("AutoPyDiscipline", py_func=g, py_jac=dgdx)

# %%
# Now, we can to minimize this :class:`.Discipline` over a design space,
# by means of a quasi-Newton method from the initial point :math:`0.5`.
#
# Define the design space
# ~~~~~~~~~~~~~~~~~~~~~~~
# For that, by means of the :func:`.create_design_space` high-level function,
# we define the :class:`.DesignSpace` :math:`[-2., 2.]`
# with initial value :math:`0.5`
# by using its :meth:`~.DesignSpace.add_variable` method.

design_space = create_design_space()
design_space.add_variable(
    "x", 1, lower_bound=-2.0, upper_bound=2.0, value=-0.5 * np.ones(1)
)

# %%
# Define the optimization problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Then, by means of the :func:`.create_scenario` high-level function,
# we define an :class:`.MDOScenario` from the :class:`.Discipline`
# and the :class:`.DesignSpace` defined above:

scenario = create_scenario(
    discipline,
    "y",
    design_space,
    formulation_name="DisciplinaryOpt",
    scenario_type="MDO",
)

# %%
# Execute the optimization problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lastly, we solve the :class:`.OptimizationProblem` included in the :class:`.MDOScenario`
# defined above by minimizing the objective function over the :class:`.DesignSpace`.
# Precisely, we choose the
# `L-BFGS-B algorithm <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_
# implemented in the function ``scipy.optimize.fmin_l_bfgs_b`` and
# indirectly called by means of the class :class:`.OptimizationLibraryFactory`
# and of its function :meth:`~.BaseAlgoFactory.execute`:

scenario.execute(algo_name="L-BFGS-B", max_iter=100)

# %%
# The optimization results are displayed in the log file. They can also be
# obtained using the following code:

optimization_result = scenario.optimization_result
print(
    f"The solution of P is (x*,f(x*)) = ({optimization_result.x_opt}, {optimization_result.f_opt})."
)

# %%
#
# .. seealso::
#
#    You can find the `SciPy <https://scipy.org/>`_ implementation of the
#    `L-BFGS-B algorithm <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_
#    algorithm `by clicking here
#    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_.
#
# In order to get the list of available optimization algorithms, use:

algo_list = get_available_opt_algorithms()
print(f"Available algorithms: {algo_list}")

# %%
# Saving and post-processing
# **************************
# After the resolution of the
# :class:`~gemseo.algos.optimization_problem.OptimizationProblem`,
# we can export the results into an :term:`HDF` file:

problem = scenario.formulation.optimization_problem
problem.to_hdf("my_optim.hdf5")

# %%
# We can also post-process the optimization history by means of the function
# :func:`.execute_post`,
# either from the :class:`~gemseo.algos.optimization_problem.OptimizationProblem`:

execute_post(problem, post_name="OptHistoryView", save=False, show=True)

# %%
# or from the :term:`HDF` file created above:

execute_post("my_optim.hdf5", post_name="OptHistoryView", save=False, show=True)
