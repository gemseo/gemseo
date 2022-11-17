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
Create a DOE Scenario
=====================

"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import get_available_doe_algorithms
from gemseo.api import get_available_post_processings

configure_logger()


# %%
#
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
# In this example, we will see how to use |g|
# to solve this problem :math:`(P)` by means of a Design Of Experiments (DOE)
#
# Define the discipline
# ---------------------
# Firstly, by means of the :meth:`~gemseo.api.create_discipline` API function,
# we create an :class:`.MDODiscipline` of :class:`.AnalyticDiscipline` type
# from a Python function:

expressions = {"y": "x1+x2"}
discipline = create_discipline("AnalyticDiscipline", expressions=expressions)

# %%
# Now, we want to minimize this :class:`.MDODiscipline`
# over a design of experiments (DOE).
#
# Define the design space
# -----------------------
# For that, by means of the :meth:`~gemseo.api.create_design_space` API function,
# we define the :class:`.DesignSpace` :math:`[-5, 5]\times[-5, 5]`
# by using its :meth:`.DesignSpace.add_variable` method.

design_space = create_design_space()
design_space.add_variable("x1", l_b=-5, u_b=5, var_type="integer")
design_space.add_variable("x2", l_b=-5, u_b=5, var_type="integer")

# %%
# Define the DOE scenario
# -----------------------
# Then, by means of the :meth:`~gemseo.api.create_scenario` API function,
# we define a :class:`.DOEScenario` from the :class:`.MDODiscipline`
# and the :class:`.DesignSpace` defined above:

scenario = create_scenario(
    discipline, "DisciplinaryOpt", "y", design_space, scenario_type="DOE"
)

# %%
# Execute the DOE scenario
# ------------------------
# Lastly, we solve the :class:`.OptimizationProblem` included in the
# :class:`.DOEScenario` defined above by minimizing the objective function
# over a design of experiments included in the :class:`.DesignSpace`.
# Precisely, we choose a `full factorial design
# <https://en.wikipedia.org/wiki/Factorial_experiment>`_ of size :math:`11^2`:

scenario.execute({"algo": "fullfact", "n_samples": 11**2})

# %%
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

# %%
# Available DOE algorithms
# ------------------------
# In order to get the list of available DOE algorithms, use:

algo_list = get_available_doe_algorithms()
print(f"Available algorithms: {algo_list}")

# %%
# Available post-processing
# -------------------------
# In order to get the list of available post-processing algorithms, use:

post_list = get_available_post_processings()
print(f"Available algorithms: {post_list}")

# %%
# You can also look at the examples:
#
# .. raw:: html
#
#    <div style="text-align: center;"><a class="btn gemseo-btn mb-1"
#    href="../post_process/index.html" role="button">Examples</a></div>
