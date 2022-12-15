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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Quadratic approximations
========================

In this example, we illustrate the use of the :class:`.QuadApprox` plot
on the Sobieski's SSBJ problem.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.sobieski.core.problem import SobieskiProblem

###############################################################################
# Import
# ------
# The first step is to import some functions from the API
# and a method to get the design space.

configure_logger()

###############################################################################
# Description
# -----------
#
# The :class:`~gemseo.post.quad_approx.QuadApprox` post-processing
# performs a quadratic approximation of a given function
# from an optimization history
# and plot the results as cuts of the approximation.

###############################################################################
# Create disciplines
# ------------------
# Then, we instantiate the disciplines of the Sobieski's SSBJ problem:
# Propulsion, Aerodynamics, Structure and Mission
disciplines = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiStructure",
        "SobieskiMission",
    ]
)

###############################################################################
# Create design space
# -------------------
# We also read the design space from the :class:`.SobieskiProblem`.
design_space = SobieskiProblem().design_space

###############################################################################
# Create and execute scenario
# ---------------------------
# The next step is to build an MDO scenario in order to maximize the range,
# encoded 'y_4', with respect to the design parameters, while satisfying the
# inequality constraints 'g_1', 'g_2' and 'g_3'. We can use the MDF formulation,
# the SLSQP optimization algorithm
# and a maximum number of iterations equal to 100.
scenario = create_scenario(
    disciplines,
    formulation="MDF",
    objective_name="y_4",
    maximize_objective=True,
    design_space=design_space,
)
scenario.set_differentiation_method()
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, "ineq")
scenario.execute({"algo": "SLSQP", "max_iter": 10})

###############################################################################
# Post-process scenario
# ---------------------
# Lastly, we post-process the scenario by means of the :class:`.QuadApprox`
# plot which performs a quadratic approximation of a given function
# from an optimization history and plot the results as cuts of the
# approximation.

###############################################################################
# .. tip::
#
#    Each post-processing method requires different inputs and offers a variety
#    of customization options. Use the API function
#    :meth:`~gemseo.api.get_post_processing_options_schema` to print a table with
#    the options for any post-processing algorithm.
#    Or refer to our dedicated page:
#    :ref:`gen_post_algos`.

###############################################################################
# The first plot shows an approximation of the Hessian matrix
# :math:`\frac{\partial^2 f}{\partial x_i \partial x_j}` based on the
# *Symmetric Rank 1* method (SR1) :cite:`Nocedal2006`. The
# color map uses a symmetric logarithmic (symlog) scale.
# This plots the cross influence of the design variables on the objective function
# or constraints. For instance, on the last figure, the maximal second-order
# sensitivity is :math:`\frac{\partial^2 -y_4}{\partial^2 x_0} = 2.10^5`,
# which means that the :math:`x_0` is the most influential variable. Then,
# the cross derivative
# :math:`\frac{\partial^2 -y_4}{\partial x_0 \partial x_2} = 5.10^4`
# is positive and relatively high compared to the previous one but the combined
# effects of :math:`x_0` and  :math:`x_2` are non-negligible in comparison.

scenario.post_process("QuadApprox", function="-y_4", save=False, show=True)

###############################################################################
# The second plot represents the quadratic approximation of the objective around the
# optimal solution : :math:`a_{i}(t)=0.5 (t-x^*_i)^2
# \frac{\partial^2 f}{\partial x_i^2} + (t-x^*_i) \frac{\partial
# f}{\partial x_i} + f(x^*)`, where :math:`x^*` is the optimal solution.
# This approximation highlights the sensitivity of the :term:`objective function`
# with respect to the :term:`design variables`: we notice that the design
# variables :math:`x\_1, x\_5, x\_6` have little influence , whereas
# :math:`x\_0, x\_2, x\_9` have a huge influence on the objective. This
# trend is also noted in the diagonal terms of the :term:`Hessian` matrix
# :math:`\frac{\partial^2 f}{\partial x_i^2}`.
