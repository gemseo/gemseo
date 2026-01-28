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
"""# Diagonal design of experiments.

Here is an illustration of the diagonal design of experiments (DOE)
implemented by the [DiagonalDOE][gemseo.algos.doe.diagonal_doe.diagonal_doe.DiagonalDOE] class
and used by the [ScalableDiagonalModel][gemseo.problems.mdo.scalable.data_driven.diagonal.ScalableDiagonalModel].
The idea is to sample the discipline by varying its inputs proportionally
on one of the diagonals of its input space.
"""

from __future__ import annotations

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix

# %%
# ## Create the discipline
#
# First, we create an [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline]
# implementing the function: $f(x)=2x-3\sin(2\pi y)$
# and set its cache policy to `"MemoryFullCache"`.

discipline = create_discipline(
    "AnalyticDiscipline", expressions={"z": "2*x-3*sin(2*pi*y)"}
)

# %%
# ## Create the design space
#
# Then, we create a [DesignSpace][gemseo.algos.design_space.DesignSpace]
# where $x$ and $y$ vary between 0 and 1.
design_space = create_design_space()
design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
design_space.add_variable("y", lower_bound=0.0, upper_bound=1.0)

# %%
# ## Sample with the default mode
#
# Lastly, we create a [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario]
# and execute it with the [DiagonalDOE][gemseo.algos.doe.diagonal_doe.diagonal_doe.DiagonalDOE] algorithm
# to get 10 evaluations of $f$.
# Note that we use the default configuration:
# all the disciplinary inputs vary proportionally
# from their lower bounds to their upper bounds.
scenario = create_scenario(
    discipline,
    "z",
    design_space,
    scenario_type="DOE",
    formulation_name="DisciplinaryOpt",
)
scenario.execute(algo_name="DiagonalDOE", n_samples=10)
dataset = scenario.to_dataset(opt_naming=False)
ScatterMatrix(dataset).execute(save=False, show=True)

# %%
# ## Sample with reverse mode for $y$
#
# We can also change the configuration
# in order to select another diagonal of the input space,
# e.g. increasing $x$ and decreasing $y$.
# This configuration is illustrated in the new [ScatterMatrix][gemseo.post.dataset.scatter_plot_matrix.ScatterMatrix] plot
# where the $(x,y)$ points follow the $t\mapsto -t$ line
# while  the $(x,y)$ points follow the $t\mapsto t$ line
# with the default configuration.
scenario = create_scenario(
    discipline,
    "z",
    design_space,
    scenario_type="DOE",
    formulation_name="DisciplinaryOpt",
)
scenario.execute(algo_name="DiagonalDOE", n_samples=10, reverse=["y"])
dataset = scenario.to_dataset(opt_naming=False)
ScatterMatrix(dataset).execute(save=False, show=True)
