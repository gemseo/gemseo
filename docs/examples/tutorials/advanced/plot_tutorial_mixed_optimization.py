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
"""# Tutorial - Solve a mixed optimization problem

## Goal

In this tutorial,
you will solve a very simple mixed optimization problem with a full factorial approach.
To do so, you will split the problem into a discrete optimization problem
to enumerate all the combinations
and a continuous one to solve the associated sub-problem.
Thus, you will use the
[MDOScenarioAdapter][gemseo.scenario.adapter.mdo_scenario_adapter.MDOScenarioAdapter]
to wrap an [MDOScenario][gemseo.scenario.mdo.MDOScenario]
and treat it as a discipline whose inputs are some or all of its design space variables
and whose outputs are some or all of its functions
(objective, constraints or observables).
Keep in mind that this approach may be very time-consuming.
It works well when the dimension of the integers to explore is not too large.
Otherwise,
a dedicated algorithm may be better suited,
such as the ones available in the `gemseo-pymoo` or `gemseo-hexaly` plugins.
"""

from __future__ import annotations

from numpy import ndarray  # noqa: TC002
from numpy.linalg import norm

from gemseo.discipline import AutoPyDiscipline
from gemseo.doe import PYDOE_FULLFACT_Settings
from gemseo.optimization import NLOPT_COBYLA_Settings
from gemseo.post import OptHistoryView_Settings
from gemseo.scenario import MDOScenario
from gemseo.scenario import MDOScenarioAdapter
from gemseo.space import DesignSpace

# %%
# ## Step 0 - Optimization problem definitions
#
# ### Initial problem
#
# You define the following optimization problem:
#
# $$
#    \begin{aligned}
#    \text{minimize the objective function }&\text{f(x,y)}=|x| + |y| \\
#    \text{with respect to the design variables }&x,\,y \\
#    \text{subject to the general constraint }
#    & g(x,y) \geq 2\\
#    \text{subject to the bound constraints }
#    & 0.0 \leq x \leq 1.0\\
#    & 0 \leq y_0 \leq 1\\
#    & 0 \leq y_2 \leq 2
#    \end{aligned}
# $$
#
# and where the general constraint is:
#
# $$g(x,y) = x + y$$
#
# Where $y$ is an integer vector with two components
# and $x$ is a float vector with two components.

# %%
# ### Problem reformulation
#
# The problem can be split using the
# [MDOScenarioAdapter][gemseo.scenario.adapter.mdo_scenario_adapter.MDOScenarioAdapter].
# To do this you will divide the design space in two,
# a continuous one and a discrete one.
# The
# [MDOScenarioAdapter][gemseo.scenario.adapter.mdo_scenario_adapter.MDOScenarioAdapter]
# will wrap the continuous inner scenario as
# a discipline to be executed taking the inputs from the discrete design space.
# These inputs are generated
# by the outer [MDOScenario][gemseo.scenario.mdo.MDOScenario]
# using a full factorial method.
# It is of course possible to use any other DOE algorithms.
#
# The reformulated optimization problem would read as follows:
# For the outer DOE Scenario:
#
# $$
#    \begin{aligned}
#    \text{minimize the objective function }&\text{f(x,y)}=|x| + |y| \\
#    \text{with respect to the design variables }&y \\
#    \text{subject to the general constraint }
#    & g(x,y) \geq 2\\
#    \text{subject to the bound constraints }
#    & 0 \leq y_0 \leq 1\\
#    & 0 \leq y_2 \leq 2
#    \end{aligned}
# $$
#
# For the inner MDO Scenario:
#
# $$
#    \begin{aligned}
#    \text{minimize the objective function }&\text{f(x,y)}=|x| + |y| \\
#    \text{with respect to the design variables }&x \\
#    \text{subject to the general constraint }
#    & g(x,y) \geq 2\\
#    \text{subject to the bound constraints }
#    & 0.0 \leq x \leq 1.0
#    \end{aligned}
# $$
#
# In the next steps, you will build both scenarios and connect them to solve the full
# problem.


# %%
# ## Step 1 - Create the discipline
#
# Since the expressions of your toy problem are very simple,
# you can use an
# [AutoPyDiscipline][gemseo.discipline.auto_py.AutoPyDiscipline]
# to compute the objective and constraints.
# Note that there are no strong couplings in your expressions,
# which means you could also compute both the objective and constraints
# with a single discipline if you wished to.
def obj(x: ndarray, y: ndarray) -> float:
    """A simple Python function to compute f(x,y).

    Args:
        x: The first operand.
        y: The second operand.

    Returns:
        The sum of the Euclidean norms of x and y.
    """
    f = norm(x) + norm(y)
    return f


def const(x: ndarray, y: ndarray) -> ndarray:
    """A simple Python function to compute g(x,y).

    Args:
        x: The first operand.
        y: The second operand.

    Returns:
        The sum of x and y.
    """
    g = x + y
    return g


objective = AutoPyDiscipline(name="f(x,y)", py_func=obj)
constraint = AutoPyDiscipline(name="g(x,y)", py_func=const)

# %%
# ## Step 2 - Create the design space for the entire problem
#
# You can define a [DesignSpace][gemseo.space.design.DesignSpace]
# for the whole problem
# and then filter either the continuous variables or the discrete ones.
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=0, upper_bound=1, value=1.0, size=2)
design_space.add_variable(
    "y", lower_bound=[0, 0], upper_bound=[1, 2], value=1, size=2, type_="integer"
)
design_space
# %%
# ## Step 3 - Create the inner scenario
#
# The inner scenario is the one that solves the continuous optimization problem,
# and as such,
# it only needs to include the continuous design variables.
# You use the
# [filter()][gemseo.space.design.DesignSpace.filter] method
# to keep `x`
# and you set `copy` to `True` to keep the original `design_space` unchanged,
# as you will use it later.
design_space_inner_scenario = design_space.filter(keep_variables=["x"], copy=True)
design_space_inner_scenario
# %%
# You then create your [MDOScenario][gemseo.scenario.mdo.MDOScenario].
# The default solver will be `NLOPT_COBYLA` with at most 100 iterations.
inner_scenario = MDOScenario([objective, constraint], design_space_inner_scenario)
inner_scenario.add_objective("f")
inner_scenario.add_constraint("g", constraint_type="ineq", value=2)
inner_scenario.set_algorithm(NLOPT_COBYLA_Settings(max_iter=100))

# %%
# ## Step 4 - Transforming into a discipline
#
# An [MDOScenarioAdapter][gemseo.scenario.adapter.mdo_scenario_adapter.MDOScenarioAdapter]
# wraps an entire [MDOScenario][gemseo.scenario.mdo.MDOScenario]
# as a [Discipline][gemseo.core.discipline.discipline.Discipline],
# its inputs are all or part of the design space variables
# and its outputs are all or part of the objective values, constraints or observables.
# Here you select the variables of the inner scenario
# that you wish to set as inputs/outputs for the adapter.
input_names = ["y"]
output_names = ["f", "g"]

# %%
# The argument `set_x0_before_opt` allows you to set the starting point
# of the adapted scenario from the outer DOE scenario values.
adapted_inner_scenario = MDOScenarioAdapter(
    inner_scenario,
    input_names,
    output_names,
    set_x0_before_opt=True,
)
# %%
#
# !!! tip
#
#     You may be interested in keeping the optimization history of the inner scenario
#     for each of the executions launched by the outer scenario.
#     To do this,
#     set the argument `keep_opt_history` to `True`,
#     this option will store the databases in memory
#     and make them accessible via the
#     [databases][gemseo.scenario.adapter.mdo_scenario_adapter.MDOScenarioAdapter.databases] attribute.
#     Keep in mind that depending on the size of the database,
#     storing it in memory may lead to a significant increase in memory usage.
#     If you prefer to store the databases on disk instead,
#     set the argument `save_opt_history` to `True`.
#     An `hdf5` file will be saved on the disk at each new execution.
#     You may also choose a prefix for the name of these files
#     with the argument `opt_history_file_prefix`.
#     If no prefix is given,
#     the default prefix is `"database"`.
#     Both `keep_opt_history` and `save_opt_history` are independent of each-other.

# %%
# ## Step 5 - Create the outer DOE scenario
#
# The outer scenario is the one that solves the discrete optimization problem,
# and as such,
# it only needs to include the integer design variables.
# Once again,
# you use the
# [filter()][gemseo.space.design.DesignSpace.filter] method to keep `y`,
# the `copy` argument ensuring that
# the original `design_space` remains unchanged
# in case you need it for other purposes.
design_space_outer_scenario = design_space.filter(keep_variables="y", copy=True)
design_space_outer_scenario
# %%
# You then create your [MDOScenario][gemseo.scenario.mdo.MDOScenario],
# and you set the same objective function.
#
outer_scenario = MDOScenario((adapted_inner_scenario,), design_space_outer_scenario)
outer_scenario.add_objective("f")

# %%
# Here, you add the constraints on the outer scenario in order to be able to know if a
# given set of integers returns a feasible solution once the inner scenario has been
# executed.
outer_scenario.add_constraint("g", constraint_type="ineq", value=2)

# %%
# ##  Step 6 - Visualize your bilevel process
#
# You can plot the xDSM, which will include both the outer and inner scenarios.
outer_scenario.xdsmize(save_html=False)

# %%
# ## Step 7 - Solve your mixed optimization problem
#
# You can execute the outer scenario
# (which contains the inner scenario)
# with a DoE algorithm,
# Doing so, you will solve your entire optimization problem.
# The console will show the progress of the optimization.
# For each DOE point,
# it will show the optimization of the continuous problem and its optimal result.
outer_scenario.execute(PYDOE_FULLFACT_Settings(n_samples=9))

# %%
#
# !!! tip
#     When using a DoE algorithm,
#     you know a priori the samples that will be evaluated.
#     This means you can run the outer scenario in parallel
#     if you set the setting `n_processes` to at least 2.
#     Note that if you are running the outer scenario in parallel
#     and requesting the databases of the continuous optimizations on the disk,
#     you will need to instantiate the
#     [MDOScenarioAdapter][gemseo.scenario.adapter.mdo_scenario_adapter.MDOScenarioAdapter]
#     with the argument `naming="UUID"`,
#     which is multiprocessing-safe.
#     Running in parallel also means that the option `keep_opt_history` will not work
#     because GEMSEO is unable to copy the databases
#     from the sub-processes to the main process.

# %%
# ## Step 8 -  Plot the objective and constraint history for the scenario
#
# At the end of the optimization you see the results of the problem. The optimal
# solution would be in this case the iteration of the DOE that gave the minimum for
# $f(x,y)$ while respecting the constraint $g(x,y)$. The full problem
# optimal result will contain the values for $x$, $y$, $f$, and
# $g$.
outer_scenario.post_process(OptHistoryView_Settings(save=False, show=True))

# %%
# ## Key takeaways
#
# You've learnt to create a bilevel scenario to tackle your mixed optimization problem.
# You used the
# [MDOScenarioAdapter][gemseo.scenario.adapter.mdo_scenario_adapter.MDOScenarioAdapter]
# to transform your continuous sub-problem into a discipline,
# so as to give it to your discrete [MDOScenario][gemseo.scenario.mdo.MDOScenario].
#
# You are aware that this technique can only help for small combinatorial problems
# (few discrete variables).
# Otherwise, you are advised to use one of the mixed optimization algorithms
# from the GEMSEO plugins instead.
