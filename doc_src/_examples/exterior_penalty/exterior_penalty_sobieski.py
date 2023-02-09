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
Example for exterior penalty applied to the Sobieski test case.
===============================================================
"""
# %%
# This section describes how to set up and solve the MDO problem relative to the
# :ref:`Sobieski test case <sobieski_problem>` with |g| applying external penalty.
#
# .. seealso::
#
#    To start with a simpler MDO problem, and for a detailed description
#    of how to plug a test case into |g|, see :ref:`sellar_mdo`.
#
#
# .. _sobieski_use_case:
#
# Solving with an :ref:`MDF formulation <mdf_formulation>`
# --------------------------------------------------------
#
# In this example, we solve the range optimization using the following
# :ref:`MDF formulation <mdf_formulation>`:
#
# - The :ref:`MDF formulation <mdf_formulation>` couples all the disciplines
#   during the :ref:`mda` at each optimization iteration.
# - All the :term:`design variables` are equally treated, concatenated in a
#   single vector and given to a single :term:`optimization algorithm` as the
#   unknowns of the problem.
# - There is no specific :term:`constraint` due to the :ref:`MDF formulation
#   <mdf_formulation>`.
# - Only the design :term:`constraints` :math:`g\_1`, :math:`g\_2` and
#   :math:`g\_3` are added to the problem.
# - The :term:`objective function` is the range (the :math:`y\_4` variable in
#   the model), computed after the :ref:`mda`.
#
# Imports
# -------
# All the imports needed for the tutorials are performed here.
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import get_available_formulations
from gemseo.disciplines.utils import get_all_inputs
from gemseo.disciplines.utils import get_all_outputs
from gemseo.problems.sobieski.core.problem import SobieskiProblem

configure_logger()

# %%
# Step 1: Creation of :class:`.MDODiscipline`
# -------------------------------------------
#
# To build the scenario, we first instantiate the disciplines. Here, the
# disciplines themselves have already been
# developed and interfaced with |g| (see :ref:`benchmark_problems`).

disciplines = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
        "SobieskiStructure",
    ]
)

# %%
# .. tip::
#
#    For the disciplines that are not interfaced with |g|, the |g|'s
#    :mod:`~gemseo.api` eases the creation of disciplines without having
#    to import them.
#
#    See :ref:`api`.

# %%
# Step 2: Creation of :class:`.Scenario`
# --------------------------------------
#
# The scenario delegates the creation of the optimization problem to the
# :ref:`MDO formulation <mdo_formulations>`.
#
# Therefore, it needs the list of :code:`disciplines`, the names of the formulation,
# the name of the objective function and the design space.
#
# - The :code:`design_space` (shown below for reference)
#   defines the unknowns of the optimization problem, and their bounds. It contains
#   all the design variables needed by the :ref:`MDF formulation <mdf_formulation>`.
#   It can be imported from a text file, or created from scratch with the methods
#   :meth:`~gemseo.api.create_design_space` and
#   :meth:`~gemseo.algos.design_space.DesignSpace.add_variable`. In this case,
#   we will retrieve it from the ``SobieskiProblem`` already defined in |g|.
design_space = SobieskiProblem().design_space
x_0 = design_space.get_current_value(as_dict=True)
# %%
#     .. code::
#
#
#           name      lower_bound      value      upper_bound  type
#           x_shared      0.01          0.05          0.09     float
#           x_shared    30000.0       45000.0       60000.0    float
#           x_shared      1.4           1.6           1.8      float
#           x_shared      2.5           5.5           8.5      float
#           x_shared      40.0          55.0          70.0     float
#           x_shared     500.0         1000.0        1500.0    float
#           x_1           0.1           0.25          0.4      float
#           x_1           0.75          1.0           1.25     float
#           x_2           0.75          1.0           1.25     float
#           x_3           0.1           0.5           1.0      float
#           y_14        24850.0    50606.9741711    77100.0    float
#           y_14        -7700.0    7306.20262124    45000.0    float
#           y_32         0.235       0.50279625      0.795     float
#           y_31         2960.0    6354.32430691    10185.0    float
#           y_24          0.44       4.15006276      11.13     float
#           y_34          0.44       1.10754577       1.98     float
#           y_23         3365.0    12194.2671934    26400.0    float
#           y_21        24850.0    50606.9741711    77250.0    float
#           y_12        24850.0      50606.9742     77250.0    float
#           y_12          0.45          0.95          1.5      float
#
# - The available :ref:`MDO formulations <mdo_formulations>` are located in the
#   **gemseo.formulations** package, see :ref:`extending-gemseo` for extending
#   GEMSEO with other formulations.
# - The :code:`formulation` classname (here, :code:`"MDF"`) shall be passed to
#   the scenario to select them.
# - The list of available formulations can be obtained by using
#   :meth:`~gemseo.api.get_available_formulations`.
get_available_formulations()
# %%
# - :math:`y\_4` corresponds to the :code:`objective_name`. This name must be one
#   of the disciplines outputs, here the ``SobieskiMission`` discipline. The list of
#   all outputs of the disciplines can be obtained by using
#   :meth:`~gemseo.disciplines.utils.get_all_outputs`:
get_all_outputs(disciplines)
get_all_inputs(disciplines)
# %%
# From these :class:`~gemseo.core.discipline.MDODiscipline`, design space,
# :ref:`MDO formulation <mdo_formulations>` name and objective function name,
# we build the scenario:
scenario = create_scenario(
    disciplines,
    formulation="MDF",
    maximize_objective=True,
    objective_name="y_4",
    design_space=design_space,
)
# %%
# The range function (:math:`y\_4`) should be maximized. However, optimizers
# minimize functions by default. Which is why, when creating the scenario, the argument
# :code:`maximize_objective` shall be set to :code:`True`.
#
# Differentiation method
# ~~~~~~~~~~~~~~~~~~~~~~
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
# The default behavior of the optimizer triggers :term:`finite differences`.
# It corresponds to:
#
# .. code::
#
#   scenario.set_differentiation_method("finite_differences",1e-7)
#
# It it also possible to differentiate functions by means of the
# :term:`complex step` method:
#
# .. code::
#
#   scenario.set_differentiation_method("complex_step",1e-30j)
#
# Constraints
# ~~~~~~~~~~~
#
# Similarly to the objective function, the constraints names are a subset
# of the disciplines' outputs. They can be obtained by using
# :meth:`~gemseo.disciplines.utils.get_all_outputs`.
#
# The formulation has a powerful feature to automatically dispatch the constraints
# (:math:`g\_1, g\_2, g\_3`) and plug them to the optimizers depending on
# the formulation. To do that, we use the method
# :meth:`~gemseo.core.scenario.Scenario.add_constraint`:
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, "ineq")
# %%
# Step 3: Apply the exterior penalty and execute the scenario
# -----------------------------------------------------------
scenario.formulation.opt_problem.apply_exterior_penalty(
    objective_scale=10.0, scale_inequality=10.0
)
# %%
# In this way the L-BFGS-B algorithm can be used to solve the optimization problem.
# Note that this algorithm is not suited for constrained optimization problems.
algo_args = {"max_iter": 10, "algo": "L-BFGS-B"}
scenario.execute(algo_args)
# %%
# Post-processing options
# ~~~~~~~~~~~~~~~~~~~~~~~
# To visualize the optimization history of the constraint violation one can use the
# :class:`.BasicHistory`:
scenario.post_process(
    "BasicHistory", variable_names=["g_1", "g_2", "g_3"], save=False, show=True
)
# %%
# This solution is almost feasible.
# The solution can better approximate the original problem solution increasing the value
#  of `objective_scale` and  `scale_inequality` parameters.
# Step 4: Rerun the scenario with increased penalty and objective scaling.
# ------------------------------------------------------------------------
design_space.set_current_value(x_0)
scenario_2 = create_scenario(
    disciplines,
    formulation="MDF",
    maximize_objective=True,
    objective_name="y_4",
    design_space=design_space,
)
for constraint in ["g_1", "g_2", "g_3"]:
    scenario_2.add_constraint(constraint, "ineq")
scenario_2.set_differentiation_method()
scenario_2.formulation.opt_problem.apply_exterior_penalty(
    objective_scale=1000.0, scale_inequality=1000.0
)
algo_args_2 = {"max_iter": 1000, "algo": "L-BFGS-B"}
scenario_2.execute(algo_args_2)
scenario_2.post_process("BasicHistory", variable_names=["-y_4"], save=False, show=True)
scenario_2.post_process(
    "BasicHistory", variable_names=["g_1", "g_2", "g_3"], save=False, show=True
)
# %%
# The solution feasibility was improved but this comes with a much higher number of
# iterations.
