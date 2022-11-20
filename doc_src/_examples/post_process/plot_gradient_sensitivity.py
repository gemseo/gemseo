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
Gradient Sensitivity
====================

In this example, we illustrate the use of the :class:`.GradientSensitivity`
plot on the Sobieski's SSBJ problem.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.sobieski.core.problem import SobieskiProblem

# %%
# Import
# ------
# The first step is to import some functions from the API
# and a method to get the design space.

configure_logger()

# %%
# Description
# -----------
#
# The :class:`.GradientSensitivity` post-processor
# builds histograms of derivatives of the objective and the constraints.

# %%
# Create disciplines
# ------------------
# At this point, we instantiate the disciplines of Sobieski's SSBJ problem:
# Propulsion, Aerodynamics, Structure and Mission
disciplines = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiStructure",
        "SobieskiMission",
    ]
)

# %%
# Create design space
# -------------------
# We also read the design space from the :class:`.SobieskiProblem`.
design_space = SobieskiProblem().design_space

# %%
# Create and execute scenario
# ---------------------------
# The next step is to build an MDO scenario in order to maximize the range,
# encoded ``"y_4"``, with respect to the design parameters, while satisfying the
# inequality constraints ``"g_1"``, ``"g_2"`` and ``"g_3"``. We can use the MDF
# formulation, the SLSQP optimization algorithm and a maximum number of iterations
# equal to 100.
scenario = create_scenario(
    disciplines,
    formulation="MDF",
    objective_name="y_4",
    maximize_objective=True,
    design_space=design_space,
)
# %%
# The differentiation method used by default is ``"user"``, which means that the
# gradient will be evaluated from the Jacobian defined in each discipline. However, some
# disciplines may not provide one, in that case, the gradient may be approximated
# with the techniques ``"finite_differences"`` or ``"complex_step"`` with the method
# :meth:`~.Scenario.set_differentiation_method`. The following line is shown as an
# example, it has no effect because it does not change the default method.
scenario.set_differentiation_method()
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, "ineq")
scenario.execute({"algo": "SLSQP", "max_iter": 10})

# %%
# Post-process scenario
# ---------------------
# Lastly, we post-process the scenario by means of the :class:`.GradientSensitivity`
# post-processor which builds histograms of derivatives of objective and constraints.
# The sensitivities shown in the plot are calculated with the gradient at the optimum
# or the least-non feasible point when the result is not feasible. One may choose any
# other iteration instead.
#
# .. note::
#    In some cases, the iteration that is being used to compute the sensitivities
#    corresponds to a point for which the algorithm did not request the evaluation of
#    the gradients, and a ``ValueError`` is raised. A way to avoid this issue is to set
#    the option ``compute_missing_gradients`` of :class:`.GradientSensitivity` to
#    ``True``, this way |g| will compute the gradients for the requested iteration if
#    they are not available.
#
# .. warning::
#    Please note that this extra computation may be expensive depending on the
#    :class:`.OptimizationProblem` defined by the user. Additionally, keep in mind that
#    |g| cannot compute missing gradients for an :class:`.OptimizationProblem` that was
#    imported from an HDF5 file.

# %%
# .. tip::
#
#    Each post-processing method requires different inputs and offers a variety
#    of customization options. Use the API function
#    :func:`.get_post_processing_options_schema` to print a table with
#    the options for any post-processing algorithm.
#    Or refer to our dedicated page:
#    :ref:`gen_post_algos`.
scenario.post_process(
    "GradientSensitivity",
    compute_missing_gradients=True,
    save=False,
    show=True,
)
