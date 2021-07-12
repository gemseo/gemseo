# -*- coding: utf-8 -*-
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
API
===

Here are some examples of the machine learning API
applied to regression models.
"""
from __future__ import division, unicode_literals

from gemseo.api import (
    configure_logger,
    create_design_space,
    create_discipline,
    create_scenario,
)
from gemseo.mlearning.api import (
    create_regression_model,
    get_regression_models,
    get_regression_options,
)

configure_logger()


###############################################################################
# Get available regression models
# -------------------------------
print(get_regression_models())

###############################################################################
# Get regression model options
# ----------------------------
print(get_regression_options("GaussianProcessRegression"))

###############################################################################
# Create regression model
# -----------------------
expressions_dict = {"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"}
discipline = create_discipline(
    "AnalyticDiscipline", name="func", expressions_dict=expressions_dict
)

design_space = create_design_space()
design_space.add_variable("x_1", l_b=0.0, u_b=1.0)
design_space.add_variable("x_2", l_b=0.0, u_b=1.0)

discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
scenario = create_scenario(
    [discipline], "DisciplinaryOpt", "y_1", design_space, scenario_type="DOE"
)
scenario.execute({"algo": "fullfact", "n_samples": 9})

dataset = discipline.cache.export_to_dataset()
model = create_regression_model("LinearRegression", data=dataset)
model.learn()

print(model)
