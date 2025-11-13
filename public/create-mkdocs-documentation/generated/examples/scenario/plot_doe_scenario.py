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
"""# Create a DOE Scenario."""

from __future__ import annotations

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import get_available_doe_algorithms
from gemseo import get_available_post_processings

# %%
#
# Let $(P)$ be a simple optimization problem:
#
# $$
#
#    (P) = \left\{
#    \begin{aligned}
#      & \underset{x\in\mathbb{N}^2}{\text{minimize}}
#      & & f(x) = x_1 + x_2 \\
#      & \text{subject to}
#      & & -5 \leq x \leq 5
#    \end{aligned}
#    \right.
# $$
#
# In this example, we will see how to use GEMSEO
# to solve this problem $(P)$ by means of a Design Of Experiments (DOE)
#
# ## Define the discipline
#
# Firstly, by means of the [create_discipline()][gemseo.create_discipline] API function,
# we create an [Discipline][gemseo.core.discipline.discipline.Discipline] of [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline] type
# from a Python function:

expressions = {"y": "x1+x2"}
discipline = create_discipline("AnalyticDiscipline", expressions=expressions)

# %%
# Now, we want to minimize this [Discipline][gemseo.core.discipline.discipline.Discipline]
# over a design of experiments (DOE).
#
# ## Define the design space
#
# For that, by means of the [create_design_space()][gemseo.create_design_space] API function,
# we define the [DesignSpace][gemseo.algos.design_space.DesignSpace] $[-5, 5]\times[-5, 5]$
# by using its [add_variable()][gemseo.algos.design_space.DesignSpace.add_variable] method.

design_space = create_design_space()
design_space.add_variable("x1", lower_bound=-5, upper_bound=5, type_="integer")
design_space.add_variable("x2", lower_bound=-5, upper_bound=5, type_="integer")

# %%
# ## Define the DOE scenario
#
# Then, by means of the [create_scenario()][gemseo.create_scenario] API function,
# we define a [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario] from the [Discipline][gemseo.core.discipline.discipline.Discipline]
# and the [DesignSpace][gemseo.algos.design_space.DesignSpace] defined above:

scenario = create_scenario(
    discipline,
    "y",
    design_space,
    scenario_type="DOE",
    formulation_name="DisciplinaryOpt",
)

# %%
# Note that the formulation settings passed to [create_scenario()][gemseo.create_scenario] can be provided
# via a Pydantic model. For more information, see [this page][formulation-settings].

# %%
# ## Execute the DOE scenario
#
# Lastly, we solve the [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] included in the
# [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario] defined above by minimizing the objective function
# over a design of experiments included in the [DesignSpace][gemseo.algos.design_space.DesignSpace].
# Precisely, we choose a [full factorial design](https://en.wikipedia.org/wiki/Factorial_experiment) of size $11^2$:

scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=11**2)

# %%
# Note that the algorithm settings passed to [execute()][gemseo.algos.base_driver_library.BaseDriverLibrary.execute] can be provided
# via a Pydantic model. For more information, see [this page][algorithm-settings].

# %%
# The optimum results can be found in the execution log. It is also possible to
# access them with [optimization_result][gemseo.scenarios.base_scenario.BaseScenario.optimization_result]:

optimization_result = scenario.optimization_result
f"The solution of P is (x*, f(x*)) = ({optimization_result.x_opt}, {optimization_result.f_opt})"

# %%
# ## Get the available DOE algorithms
#
# In order to get the list of available DOE algorithms, use:
get_available_doe_algorithms()

# %%
# ## Get the available post-processing
#
# In order to get the list of available post-processing algorithms, use:
get_available_post_processings()

# %%
# You can also look at the examples:
#
# .. raw:: html
#
#    <div style="text-align: center;"><a class="btn gemseo-btn mb-1"
#    href="../post_process/index.html" role="button">Examples</a></div>
