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
#        :author: Arthur Piat, Francois Gallard
#
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Multi-start optimization
========================

The optimization algorithm ``multistart``
generates starting points using a DOE algorithm
and run a sub-optimization algorithm from each starting point.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import execute_post
from gemseo.algos.opt.multi_start.settings.multi_start_settings import (
    MultiStart_Settings,
)

configure_logger()


# %%
# First,
# we create the disciplines
objective = create_discipline("AnalyticDiscipline", expressions={"obj": "x**3-x+1"})
constraint = create_discipline(
    "AnalyticDiscipline", expressions={"cstr": "x**2+obj**2-1.5"}
)

# %%
# and the design space
design_space = create_design_space()
design_space.add_variable("x", lower_bound=-1.5, upper_bound=1.5, value=1.5)

# %%
# Then,
# we define the MDO scenario
scenario = create_scenario(
    [objective, constraint],
    "obj",
    design_space,
    formulation_name="DisciplinaryOpt",
)
# %%
# Note that the formulation settings passed to :func:`.create_scenario` can be provided
# via a Pydantic model. For more information, see :ref:`formulation_settings`.

scenario.add_constraint("cstr", constraint_type="ineq")

# %%
# and execute it with the ``MultiStart`` optimization algorithm
# combining the local optimization algorithm SLSQP
# and the full-factorial DOE algorithm:
multistart_settings = MultiStart_Settings(
    max_iter=100,
    opt_algo_name="SLSQP",
    doe_algo_name="PYDOE_FULLFACT",
    n_start=10,
    # Set multistart_file_path to save the history of the local optima.
    multistart_file_path="multistart.hdf5",
)
scenario.execute(multistart_settings)

# %%
# Lastly,
# we can plot the history of the objective,
# either by concatenating the 10 sub-optimization histories:
execute_post(
    scenario, post_name="BasicHistory", variable_names=["obj"], save=False, show=True
)

# %%
# or by filtering the local optima (one per starting point):
execute_post(
    "multistart.hdf5",
    post_name="BasicHistory",
    variable_names=["obj"],
    save=False,
    show=True,
)
