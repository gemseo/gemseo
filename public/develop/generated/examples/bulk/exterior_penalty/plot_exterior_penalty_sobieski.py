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

"""# Example for exterior penalty applied to the Sobieski test case."""

# %%
# This section describes how to set up and solve the MDO problem relative to the
# [Sobieski's SSBJ problem][sobieskis-ssbj-test-case] with GEMSEO applying external penalty.
#
# !!! info "See also"
#
#     To start with a simpler MDO problem, and for a detailed description
#     of how to plug a test case into GEMSEO, see
#     [this example][a-from-scratch-example-on-the-sellar-problem].
#
# ## Solving with an [MDF formulation][the-mdf-formulation]
#
# In this example, we solve the range optimization using the following
# [MDF formulation][the-mdf-formulation]:
#
# - The [MDF formulation][the-mdf-formulation] couples all the disciplines
#   during the [MDA][multi-disciplinary-analyses] at each optimization iteration.
# - All the design variables are equally treated, concatenated in a
#   single vector and given to a single optimization algorithm as the
#   unknowns of the problem.
# - There is no specific constraint due to the [MDF formulation][the-mdf-formulation].
# - Only the design constraints $g_1$, $g_2$ and
#   $g_3$ are added to the problem.
# - The objective function is the range (the $y_4$ variable in
#   the model), computed after the [MDA][multi-disciplinary-analyses].
#
# ## Imports
#
# All the imports needed for the tutorials are performed here.
from __future__ import annotations

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import get_available_formulations
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.settings.formulations import MDF_Settings
from gemseo.utils.discipline import get_all_inputs
from gemseo.utils.discipline import get_all_outputs

# %%
# ## Step 1: Creation of [Discipline][gemseo.core.discipline.discipline.Discipline]
#
# To build the scenario, we first instantiate the disciplines. Here, the
# disciplines themselves have already been
# developed and interfaced with GEMSEO (see [the benchmark problem][benchmark-problems]).

disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])

# %%
# !!! tip
#
#     For the disciplines that are not interfaced with GEMSEO, the GEMSEO's
#     [gemseo][gemseo] eases the creation of disciplines without having
#     to import them.
#
#     See :ref:`api`.

# %%
# ## Step 2: Creation of [BaseScenario][gemseo.scenarios.base_scenario.BaseScenario]
#
# The scenario delegates the creation of the optimization problem to the
# [MDO formulation][mdo-formulations].
#
# Therefore, it needs the list of `disciplines`, the name of the formulation
# (or its settings model), the name of the objective function and the design space.
#
# - The `design_space` (shown below for reference)
#   defines the unknowns of the optimization problem, and their bounds. It contains
#   all the design variables needed by the [MDF formulation][the-mdf-formulation].
#   It can be imported from a text file, or created from scratch with the methods
#   [create_design_space()][gemseo.create_design_space] and
#   [add_variable()][gemseo.algos.design_space.DesignSpace.add_variable]. In this case,
#   we will use `SobieskiDesignSpace` already defined in GEMSEO.
design_space = SobieskiDesignSpace()
x_0 = design_space.get_current_value(as_dict=True)
# %%
# ```
# name      lower_bound      value      upper_bound  type
# x_shared      0.01          0.05          0.09     float
# x_shared    30000.0       45000.0       60000.0    float
# x_shared      1.4           1.6           1.8      float
# x_shared      2.5           5.5           8.5      float
# x_shared      40.0          55.0          70.0     float
# x_shared     500.0         1000.0        1500.0    float
# x_1           0.1           0.25          0.4      float
# x_1           0.75          1.0           1.25     float
# x_2           0.75          1.0           1.25     float
# x_3           0.1           0.5           1.0      float
# y_14        24850.0    50606.9741711    77100.0    float
# y_14        -7700.0    7306.20262124    45000.0    float
# y_32         0.235       0.50279625      0.795     float
# y_31         2960.0    6354.32430691    10185.0    float
# y_24          0.44       4.15006276      11.13     float
# y_34          0.44       1.10754577       1.98     float
# y_23         3365.0    12194.2671934    26400.0    float
# y_21        24850.0    50606.9741711    77250.0    float
# y_12        24850.0      50606.9742     77250.0    float
# y_12          0.45          0.95          1.5      float
# ```
#
# - The available [MDO formulations][mdo-formulations] are located in the
#   **gemseo.formulations** package, see [this page][extend-gemseo-features] for extending
#   GEMSEO with other formulations.
# - The `formulation` name (here, `"MDF"`) shall be passed to
#   the scenario to select them.
# - The list of available formulations can be obtained by using
#   [get_available_formulations()][gemseo.get_available_formulations].
get_available_formulations()
# %%
# - $y_4$ corresponds to the `objective_name`. This name must be one
#   of the disciplines outputs, here the `SobieskiMission` discipline. The list of
#   all outputs of the disciplines can be obtained by using
#   [get_all_outputs()][gemseo.utils.discipline.get_all_outputs]:
get_all_outputs(disciplines)
get_all_inputs(disciplines)
# %%
# From these [Discipline][gemseo.core.discipline.discipline.Discipline], design space,
# [MDO formulation][mdo-formulations] name and objective function name,
# we build the scenario:
scenario = create_scenario(
    disciplines,
    "y_4",
    design_space,
    maximize_objective=True,
    formulation_name="MDF",
)

# %%
# Note that both the formulation settings passed to [create_scenario()][gemseo.create_scenario] can be provided
# via a Pydantic model. For more information, see [this page][formulation-settings].

# %%
# The range function ($y_4$) should be maximized. However, optimizers
# minimize functions by default. Which is why, when creating the scenario, the argument
# `maximize_objective` shall be set to `True`.
#
# ### Differentiation method
#
# We may choose the way derivatives are computed:
#
#
# **Function derivatives.** As analytical disciplinary derivatives are
# available for the Sobieski test-case, they can be used instead of computing
# the derivatives with finite-differences or with the complex-step method.
# The easiest way to set a method is to let the optimizer determine it:
scenario.set_differentiation_method()
# %%
#
# The default behavior of the optimizer triggers finite differences.
# It corresponds to:
#
# ``` python
# scenario.set_differentiation_method("finite_differences",1e-7)
# ```
#
# It is also possible to differentiate functions by means of the
# complex step method:
#
# ``` python
# scenario.set_differentiation_method("complex_step",1e-30j)
# ```
#
# ### Constraints
#
# Similarly to the objective function, the constraints names are a subset
# of the disciplines' outputs. They can be obtained by using
# [get_all_outputs()][gemseo.utils.discipline.get_all_outputs].
#
# The formulation has a powerful feature to automatically dispatch the constraints
# ($g_1, g\_2, g\_3$) and plug them to the optimizers depending on
# the formulation. To do that, we use the method
# [add_constraint()][gemseo.scenarios.base_scenario.BaseScenario.add_constraint]:
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, constraint_type="ineq")
# %%
# ## Step 3: Apply the exterior penalty and execute the scenario
scenario.formulation.optimization_problem.apply_exterior_penalty(
    objective_scale=10.0, scale_inequality=10.0
)
# %%
# In this way the L-BFGS-B algorithm can be used to solve the optimization problem.
# Note that this algorithm is not suited for constrained optimization problems.
scenario.execute(algo_name="L-BFGS-B", max_iter=10)

# %%
# Note that the algorithm settings passed to
# [execute()][gemseo.scenarios.base_scenario.BaseScenario.execute] can be provided via a
# Pydantic model. For more information, [this page][algorithm-settings].

# %%
# ### Post-processing options
#
# To visualize the optimization history of the constraint violation one can use the
# [BasicHistory][gemseo.post.basic_history.BasicHistory]:
scenario.post_process(
    post_name="BasicHistory",
    variable_names=["g_1", "g_2", "g_3"],
    save=False,
    show=True,
)
# %%
# This solution is almost feasible.
# The solution can better approximate the original problem solution increasing the value
# of `objective_scale` and  `scale_inequality` parameters.
#
# ## Step 4: Rerun the scenario with increased penalty and objective scaling.
design_space.set_current_value(x_0)

# %%
# Here, we use the [MDF_Settings][gemseo.formulations.mdf_settings.MDF_Settings] model
# to define the formulation instead of passing the settings one by one as we did the first time.
# Both ways of defining the settings are equally valid.

scenario_2 = create_scenario(
    disciplines,
    "y_4",
    design_space,
    formulation_settings_model=MDF_Settings(),
    maximize_objective=True,
)
for constraint in ["g_1", "g_2", "g_3"]:
    scenario_2.add_constraint(constraint, constraint_type="ineq")
scenario_2.set_differentiation_method()
scenario_2.formulation.optimization_problem.apply_exterior_penalty(
    objective_scale=1000.0, scale_inequality=1000.0
)

# %%
# Here, we use the [L_BFGS_B_Settings][gemseo.algos.opt.scipy_local.settings.lbfgsb.L_BFGS_B_Settings] model
# to define the algorithm settings instead of
# passing them one by one as we did the first time. Both ways of defining the settings
# are equally valid.
from gemseo.settings.opt import L_BFGS_B_Settings  # noqa: E402

scenario_2.execute(L_BFGS_B_Settings(max_iter=1000))
scenario_2.post_process(
    post_name="BasicHistory", variable_names=["-y_4"], save=False, show=True
)

# %%
# Here, we use the [BasicHistory_Settings][gemseo.post.basic_history_settings.BasicHistory_Settings] model
# to define the post-processor settings
# instead of passing them one by one as we did the first time.
# Both ways of defining the settings are equally valid.
from gemseo.settings.post import BasicHistory_Settings  # noqa: E402

scenario_2.post_process(
    BasicHistory_Settings(
        variable_names=["g_1", "g_2", "g_3"],
        save=False,
        show=True,
    )
)
# %%
# The solution feasibility was improved but this comes with a much higher number of
# iterations.
