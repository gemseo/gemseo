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
Diagonal design of experiments
==============================

Here is an illustration of the diagonal design of experiments (DOE)
implemented by the :class:`.DiagonalDOE` class
and used by the :class:`.ScalableDiagonalModel`.
The idea is to sample the discipline by varying its inputs proportionally
on one of the diagonals of its input space.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario

configure_logger()


###############################################################################
# Create the discipline
# ---------------------
# First, we create an :class:`.AnalyticDiscipline`
# implementing the function: :math:`f(x)=2x-3\sin(2\pi y)`
# and set its cache policy to :code:`"MemoryFullCache"`.

discipline = create_discipline(
    "AnalyticDiscipline", expressions={"z": "2*x-3*sin(2*pi*y)"}
)

###############################################################################
# Create the design space
# -----------------------
# Then, we create a :class:`.DesignSpace`
# where :math:`x` and :math:`y` vary between 0 and 1.
design_space = create_design_space()
design_space.add_variable("x", l_b=0.0, u_b=1.0)
design_space.add_variable("y", l_b=0.0, u_b=1.0)

###############################################################################
# Sample with the default mode
# ----------------------------
# Lastly, we create a :class:`.DOEScenario`
# and execute it with the :class:`.DiagonalDOE` algorithm
# to get 10 evaluations of :math:`f`.
# Note that we use the default configuration:
# all the disciplinary inputs vary proportionally
# from their lower bounds to their upper bounds.
scenario = create_scenario(
    discipline, "DisciplinaryOpt", "z", design_space, scenario_type="DOE"
)
scenario.execute({"algo": "DiagonalDOE", "n_samples": 10})

dataset = scenario.export_to_dataset(opt_naming=False)
dataset.plot("ScatterMatrix", save=False, show=True)

###############################################################################
# Sample with reverse mode for :math:`y`
# --------------------------------------
# We can also change the configuration
# in order to select another diagonal of the input space,
# e.g. increasing :math:`x` and decreasing :math:`y`.
# This configuration is illustrated in the new :class:`.ScatterMatrix` plot
# where the :math:`(x,y)` points follow the :math:`t\mapsto -t` line
# while  the :math:`(x,y)` points follow the :math:`t\mapsto t` line
# with the default configuration.
scenario = create_scenario(
    discipline, "DisciplinaryOpt", "z", design_space, scenario_type="DOE"
)
scenario.execute(
    {"algo": "DiagonalDOE", "n_samples": 10, "algo_options": {"reverse": ["y"]}}
)

dataset = scenario.export_to_dataset(opt_naming=False)
dataset.plot("ScatterMatrix", save=False, show=True)
