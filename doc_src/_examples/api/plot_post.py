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
Post-processing
===============

In this example, we will discover the different functions of the API
related to graphical post-processing of scenarios.

"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import execute_post
from gemseo import get_available_post_processings
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace

configure_logger()


# %%
# Get available post-processors
# -----------------------------
#
# The :func:`.get_available_post_processings` function returns the list
# of post-processing algorithms available in |g| or in external modules
get_available_post_processings()

# %%
# Post-process a scenario
# -----------------------
# The API function :func:`.execute_post` can generate visualizations
# of the optimization or DOE results. For that, it considers the object to
# post-process ``to_post_proc``, the post-processor ``post_name``
# with its ``**settings``. E.g.
disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
design_space = SellarDesignSpace()
scenario = create_scenario(
    disciplines, "obj", design_space, name="SellarMDFScenario", formulation_name="MDF"
)
scenario.add_constraint("c_1", constraint_type="ineq")
scenario.add_constraint("c_2", constraint_type="ineq")
scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)
execute_post(scenario, post_name="OptHistoryView", save=False, show=False)

# %%
# It is also possible to pass a settings model to :func:`.execute_post` with the keyword
# ``settings_model``, as shown below. See :ref:`post_processor_settings` for more
# information.
from gemseo.settings.post import ConstraintsHistory_Settings  # noqa: E402

execute_post(
    scenario,
    settings_model=ConstraintsHistory_Settings(
        constraint_names=["c_1", "c_2"],
        save=False,
        show=True,
    ),
)
