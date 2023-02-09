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
#        :author: Francois Gallard, Damien Guénot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Application: Sobieski's Super-Sonic Business Jet (MDO)
======================================================
"""
##############################################################################
# This section describes how to setup and solve the MDO problem relative to the
# :ref:`Sobieski test case <sobieski_problem>` with |g|.
#
# .. seealso::
#
#    To begin with a more simple MDO problem, and have a detailed description
#    of how to plug a test case to |g|, see :ref:`sellar_mdo`.
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
# Note that some of the imports are related to the Python 2/3 compatibility.
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import get_available_formulations
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.disciplines.utils import get_all_inputs
from gemseo.disciplines.utils import get_all_outputs
from gemseo.problems.sobieski.core.problem import SobieskiProblem

configure_logger()

##############################################################################
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

##############################################################################
# .. tip::
#
#    For the disciplines that are not interfaced with |g|, the |g|'s
#    :mod:`~gemseo.api` eases the creation of disciplines without having
#    to import them.
#
#    See :ref:`api`.

##############################################################################
# Step 2: Creation of :class:`.Scenario`
# --------------------------------------
#
# The scenario delegates the creation of the optimization problem to the
# :ref:`MDO formulation <mdo_formulations>`.
#
# Therefore, it needs the list of :code:`disciplines`, the names of the formulation,
# the name of the objective function and the design space.
#
# - The :code:`design_space` (shown below for reference, as :code:`design_space.txt`)
#   defines the unknowns of the optimization problem, and their bounds. It contains
#   all the design variables needed by the :ref:`MDF formulation <mdf_formulation>`.
#   It can be imported from a text file, or created from scratch with the methods
#   :meth:`~gemseo.api.create_design_space` and
#   :meth:`~gemseo.algos.design_space.DesignSpace.add_variable`. In this case,
#   we will create it directly from the API.
design_space = SobieskiProblem().design_space
##############################################################################
#     .. code::
#
#           vi design_space.txt
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
##############################################################################
# - :math:`y\_4` corresponds to the :code:`objective_name`. This name must be one
#   of the disciplines outputs, here the "SobieskiMission" discipline. The list of
#   all outputs of the disciplines can be obtained by using
#   :meth:`~gemseo.disciplines.utils.get_all_outputs`:
get_all_outputs(disciplines)
get_all_inputs(disciplines)
##############################################################################
# From these :class:`~gemseo.core.discipline.MDODiscipline`, design space filename,
# :ref:`MDO formulation <mdo_formulations>` name and objective function name,
# we build the scenario:
scenario = create_scenario(
    disciplines,
    formulation="MDF",
    maximize_objective=True,
    objective_name="y_4",
    design_space=design_space,
)
##############################################################################
# The range function (:math:`y\_4`) should be maximized. However, optimizers
# minimize functions by default. Which is why, when creating the scenario, the argument
# :code:`maximize_objective` shall be set to :code:`True`.
#
# Scenario options
# ~~~~~~~~~~~~~~~~
#
# We may provide additional options to the scenario:
#
#
# **Function derivatives.** As analytical disciplinary derivatives are
# vailable for Sobieski test-case, they can be used instead of computing
# the derivatives with finite-differences or with the complex-step method.
# The easiest way to set a method is to let the optimizer determine it:
scenario.set_differentiation_method()
##############################################################################
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
##############################################################################
# Step 3: Execution and visualization of the results
# --------------------------------------------------
#
# The algorithm arguments are provided as a dictionary to the execution
# method of the scenario:
algo_args = {"max_iter": 10, "algo": "SLSQP"}
##############################################################################
# .. warning::
#
#    The mandatory arguments are the maximum number of iterations and the algorithm name.
#    Any other options of the optimization algorithm can be prescribed through
#    the argument :code:`algo_options` with a dictionary, e.g.
#    :code:`algo_args =
#    {"max_iter": 10, "algo": "SLSQP": "algo_options": {"ftol_rel": 1e-6}}`.
#    This list of available algorithm options are detailed here: :ref:`gen_opt_algos`.
#
# The scenario is executed by means of the line:
scenario.execute(algo_args)
##############################################################################
# Post-processing options
# ~~~~~~~~~~~~~~~~~~~~~~~
# A whole variety of visualizations may be displayed for both MDO and DOE
# scenarios. These features are illustrated on the SSBJ use case in
# :ref:`post_processing`.
#
# To visualize the optimization history:
scenario.post_process("OptHistoryView", save=False, show=True)

##############################################################################
# Influence of gradient computation method on performance
# -------------------------------------------------------
#
# As mentioned in :ref:`jacobian_assembly`, several methods
# are available in order to perform  the gradient computations: classical finite
# differences, complex step and :ref:`mda` linearization in direct or adjoint mode.
# These modes are automatically selected by |g| to minimize the CPU time. Yet, they
# can be forced on demand in each :ref:`mda`:
scenario.formulation.mda.linearization_mode = JacobianAssembly.DIRECT_MODE
scenario.formulation.mda.matrix_type = JacobianAssembly.LINEAR_OPERATOR
##############################################################################
# The method used to solve the adjoint or direct linear problem may also be selected.
# |g| can either assemble a sparse residual Jacobian matrix of the :ref:`mda` from the
# disciplines matrices. This has the advantage that LU factorizations may be stored to
# solve multiple right hand sides problems in a cheap way. But this requires
# extra memory.
scenario.formulation.mda.matrix_type = JacobianAssembly.SPARSE
scenario.formulation.mda.use_lu_fact = True
##############################################################################
# Altenatively, |g| can implicitly create a matrix-vector product operator,
# which is sufficient for GMRES-like solvers. It avoids to create an additional
# data structure. This can also be mandatory if the disciplines do not provide
# full Jacobian matrices but only matrix-vector product operators.
scenario.formulation.mda.matrix_type = JacobianAssembly.LINEAR_OPERATOR
##############################################################################
# The next table shows the performance of each method for solving the Sobieski use case
# with :ref:`MDF <mdf_formulation>` and :ref:`IDF <idf_formulation>` formulations.
# Efficiency of linearization is clearly visible has it takes from 10 to 20 times
# less CPU time to compute analytic derivatives of an :ref:`mda` compared
# to finite difference and complex step.
# For :ref:`IDF <idf_formulation>`, improvements are less consequent,
# but direct linearization is more than 2.5 times faster than other methods.
#
# .. tabularcolumns:: |l|c|c|
#
# +-----------------------+------------------------------+------------------------------+
# |                       |  Execution time (s)                                         |
# +  Derivation Method    +------------------------------+------------------------------+
# |                       | :ref:`MDF <mdf_formulation>` | :ref:`IDF <idf_formulation>` |
# +=======================+==============================+==============================+
# | Finite differences    | 8.22                         | 1.93                         |
# +-----------------------+------------------------------+------------------------------+
# | Complex step          | 18.11                        | 2.07                         |
# +-----------------------+------------------------------+------------------------------+
# | Linearized (direct)   | 0.90                         | 0.68                         |
# +-----------------------+------------------------------+------------------------------+
