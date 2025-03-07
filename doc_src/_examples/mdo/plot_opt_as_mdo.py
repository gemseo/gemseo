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
"""
Make a monodisciplinary optimization problem multidisciplinary
==============================================================

Introduction
------------

The :class:`.OptAsMDOScenario` is a monodisciplinary optimization scenario made multidisciplinary.
The only requirement is that
the discipline has at least three scalar inputs and at least one output.
This scenario can be used to enrich a catalog of benchmark MDO problems,
based on the observation that
MDO benchmark problems are far less numerous than optimization problems.

This example illustrates it
in the case of the minimization of the 3-dimensional Rosenbrock function

.. math::

   f(z) = 100(z_2-z_1^2)^2 + (1-z_1)^2 + 100(z_1-z_0^2)^2 + (1-z_0)^2

over the hypercube :math:`[-1,1]^3`.

The unique solution of this minimization problem is
the design point :math:`z^*=(1,1,1)` at which :math:`f` is zero.
"""

from __future__ import annotations

from numpy import array

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import generate_coupling_graph
from gemseo.problems.mdo.opt_as_mdo_scenario import OptAsMDOScenario

configure_logger()

# %%
# Material
# --------
# First,
# we create the discipline implementing the Rosenbrock function:
discipline = create_discipline(
    "AnalyticDiscipline",
    expressions={"f": "100*(z_2-z_1**2)**2+(1-z_1)**2+100*(z_1-z_0**2)**2+(1-z_0)**2"},
    name="Rosenbrock",
)
# %%
# as well as the design space:
design_space = create_design_space()
design_space.add_variable("z_0", lower_bound=-1, upper_bound=1)
design_space.add_variable("z_1", lower_bound=-1, upper_bound=1)
design_space.add_variable("z_2", lower_bound=-1, upper_bound=1)
# %%
# and choose :math:`x^{(0)}=(-0.25, 0.75, -0.9)`
# as the starting point of the optimization:
initial_point = array([-0.25, 0.75, -0.9])
design_space.set_current_value(initial_point)

# %%
# Optimization problem
# --------------------
# Then,
# we define the optimization problem:
opt_scenario = create_scenario(
    [discipline], "f", design_space, formulation_name="DisciplinaryOpt"
)
# %%
# and solve it using the SLSQP algorithm:
opt_scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)
# %%
# We can see that the numerical solution corresponds to the analytical one.
#
# MDO problem
# -----------
# Now,
# we use the :class:`.OptAsMDOScenario` to rewrite this optimization problem
# as an MDO problem with two strongly coupled disciplines.
#
# First,
# we reset the design space to the initial point:
design_space.set_current_value(initial_point)
# %%
# and create the :class:`.OptAsMDOScenario`,
# orchestrated by an MDF formulation:
mdo_scenario = OptAsMDOScenario(discipline, "f", design_space, formulation_name="MDF")
# %%
# Then,
# we can see that the design variables have been renamed:
design_space
# %%
# This renaming is based on the convention:
#
# - the first design variable is the global design variable and is named :math:`x_0`,
# - the :math:`(1+i)`-th design variable is the local design variable
#   specific to the :math:`i`-th strongly coupled discipline
#   and is named :math:`x_{1+i}`.
#
# We can also have a look to the coupling graph:
generate_coupling_graph(mdo_scenario.disciplines, file_path="")
# %%
# and see that there are two strongly coupled disciplines :math:`D_1` and :math:`D_2`,
# connected by the coupling variables :math:`y_1` and :math:`y_2`.
# These disciplines are weakly coupled to a downstream link discipline :math:`L`,
# which is weakly coupled to the downstream original discipline.
# Let us note that the link discipline computes
# the values of the design variables in the original optimization problem
# from the values of the design and coupling variables in the MDO problem.
#
# Lastly,
# we solve this scenario using the SLSQP algorithm:
mdo_scenario.set_differentiation_method(method="finite_differences")
mdo_scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)
# %%
# We can see that the numerical solution corresponds to the analytical one.
