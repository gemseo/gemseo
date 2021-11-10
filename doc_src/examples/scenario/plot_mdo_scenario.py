# -*- coding: utf-8 -*-
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
Create an MDO Scenario
======================

"""
from __future__ import division, unicode_literals

from numpy import ones

from gemseo.api import (
    configure_logger,
    create_design_space,
    create_discipline,
    create_scenario,
    get_available_opt_algorithms,
    get_available_post_processings,
)

configure_logger()


#############################################################################
# Let :math:`(P)` be a simple optimization problem:
#
# .. math::
#
#   (P) = \left\{
#   \begin{aligned}
#     & \underset{x}{\text{minimize}}
#     & & f(x) = \sin(x) - \exp(x) \\
#     & \text{subject to}
#     & & -2 \leq x \leq 2
#   \end{aligned}
#   \right.
#
# In this subsection, we will see how to use |g| to solve this problem
# :math:`(P)` by means of an optimization algorithm.
#
# Define the discipline
# ---------------------
# Firstly, by means of the :meth:`~gemseo.api.create_discipline` API function,
# we create an :class:`.MDODiscipline` of :class:`.AnalyticDiscipline` type
# from a Python function:

expressions_dict = {"y": "sin(x)-exp(x)"}
discipline = create_discipline("AnalyticDiscipline", expressions_dict=expressions_dict)

#############################################################################
# Now, we can to minimize this :class:`.MDODiscipline` over a design space,
# by means of a quasi-Newton method from the initial point :math:`0.5`.
#
# Define the design space
# -----------------------
# For that, by means of the :meth:`~gemseo.api.create_design_space` API function,
# we define the :class:`.DesignSpace` :math:`[-2, 2]` with initial value :math:`0.5`
# by using its :meth:`.DesignSpace.add_variable` method.

design_space = create_design_space()
design_space.add_variable("x", 1, l_b=-2.0, u_b=2.0, value=-0.5 * ones(1))

#############################################################################
# Define the MDO scenario
# -----------------------
# Then, by means of the :meth:`~gemseo.api.create_scenario` API function,
# we define an :class:`.MDOScenario` from the :class:`.MDODiscipline`
# and the :class:`.DesignSpace` defined above:

scenario = create_scenario(
    discipline, "DisciplinaryOpt", "y", design_space, scenario_type="MDO"
)

#############################################################################
# What about the differentiation method?
# --------------------------------------
# The :class:`.AnalyticDiscipline` automatically differentiates the
# expressions to obtain the Jacobian matrices. Therefore, there is no need to
# define a differentiation method in this case. Keep in mind that for a
# generic discipline with no defined Jacobian function, you can use the
# :meth:`.Scenario.set_differentiation_method` method to define a numerical
# approximation of the gradients.
#
# .. code::
#
#    scenario.set_differentiation_method('user')

#############################################################################
# Execute the MDO scenario
# ------------------------
# Lastly, we solve the :class:`.OptimizationProblem` included in the
# :class:`.MDOScenario` defined above by minimizing the objective function over
# the :class:`.DesignSpace`. Precisely, we choose the `L-BFGS-B algorithm
# <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_ implemented in the
# function :code:`scipy.optimize.fmin_l_bfgs_b`.

scenario.execute({"algo": "L-BFGS-B", "max_iter": 100})

#############################################################################
# The optimum results can be found in the execution log. It is also possible to
# extract them by invoking the :meth:`.Scenario.get_optimum` method. It
# returns a dictionary containing the optimum results for the
# scenario under consideration:

opt_results = scenario.get_optimum()
print(
    "The solution of P is (x*,f(x*)) = ({}, {})".format(
        opt_results.x_opt, opt_results.f_opt
    ),
)

#############################################################################
# .. seealso::
#
#    You can found the `scipy <https://www.scipy.org/>`_ implementation of the
#    `L-BFGS-B algorithm <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_
#    algorithm `by clicking here
#    <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_.  # noqa
#
# Available algorithms
# --------------------
# In order to get the list of available optimization algorithms, use:
algo_list = get_available_opt_algorithms()
print("Available algorithms: {}".format(algo_list))

#############################################################################
# Available post-processing
# -------------------------
# In order to get the list of available post-processing algorithms, use:
post_list = get_available_post_processings()
print("Available algorithms: {}".format(post_list))
#############################################################################
# You can also look at the examples:
#
# .. raw:: html
#
#    <div style="text-align: center;"><a class="btn gemseo-btn mb-1"
#    href="../post_process/index.html" role="button">Examples</a></div>
