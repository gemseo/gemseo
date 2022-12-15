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

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import execute_post
from gemseo.api import get_available_post_processings
from gemseo.api import get_post_processing_options_schema
from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace

configure_logger()


##############################################################################
# Get available DOE algorithms
# ----------------------------
#
# The :meth:`~gemseo.api.get_available_post_processings` function returns the list
# of post-processing algorithms available in |g| or in external modules
print(get_available_post_processings())

##########################################################################
# Get options schema
# ------------------
# For a given post-processing algorithm, e.g. :code:`"RadarChart"`,
# we can get the options; e.g.
print(get_post_processing_options_schema("RadarChart"))

##########################################################################
# Post-process a scenario
# -----------------------
# The API function :meth:`~gemseo.api.execute_post` can generate visualizations
# of the optimization or DOE results. For that, it consider the object to
# post-process :code:`to_post_proc`, the post processing :code:`post_name`
# with its :code:`**options`. E.g.
disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
design_space = SellarDesignSpace()
scenario = create_scenario(disciplines, "MDF", "obj", design_space, "SellarMDFScenario")
scenario.execute({"algo": "NLOPT_SLSQP", "max_iter": 100})
execute_post(scenario, "OptHistoryView", save=False, show=True)
