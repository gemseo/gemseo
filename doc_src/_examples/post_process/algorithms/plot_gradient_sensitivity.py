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
post-processing on the Sobieski's SSBJ problem.

The :class:`.GradientSensitivity` post-processor plots histograms of the objective and
the constraints derivatives.

By default, the sensitivities are calculated either at the optimum, or when the result
is not feasible, at the least-non feasible point. The iteration where the sensitivities
are computed can be modified via the `iteration` setting.

.. note::
  In some cases, the iteration used to compute the sensitivities corresponds to a
  point for which the algorithm did not request the evaluation of the gradients. In
  this case, a `ValueError` is raised by :class:`.GradientSensitivity`.
  To overcome this issue, one can set the `compute_missing_gradients` setting to True.
  This way, |g| will compute the gradients for the iterations where it is lacking.
  This can be done only if the underlying disciplines are available, explaing why
  why, unlike the other post-processing examples, we need to **post-process directly
  from the MDO scenario**.

.. warning::
   Please note that this extra computation may be expensive depending on the
   :class:`.OptimizationProblem` defined by the user. Additionally, keep in mind that
   |g| cannot compute missing gradients for an :class:`.OptimizationProblem` that was
   imported from an HDF5 file.
"""

# %%
# MDO scenario
# ------------
#
# Let us first create and execute the MDF scenario on the Sobieski's SSBJ problem.
from __future__ import annotations

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.settings.post import GradientSensitivity_Settings

disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])

formulation_settings = MDF_Settings(
    main_mda_name="MDAGaussSeidel",
    main_mda_settings=MDAGaussSeidel_Settings(
        max_mda_iter=30,
        tolerance=1e-10,
        warm_start=True,
        use_lu_fact=True,
    ),
)

scenario = create_scenario(
    disciplines,
    "y_4",
    design_space=SobieskiDesignSpace(),
    maximize_objective=True,
    formulation_settings_model=formulation_settings,
)

for name in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(name, constraint_type="ineq")

scenario.execute(SLSQP_Settings(max_iter=20))


# %%
# Post-processing
# ---------------
# Let us now post-process the scenario by means of the :class:`.GradientSensitivity`.

scenario.post_process(
    settings_model=GradientSensitivity_Settings(
        compute_missing_gradients=True,
        save=False,
        show=True,
    ),
)
