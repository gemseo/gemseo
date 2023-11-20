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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Change the seed of a DOE
========================

`Latin hypercube sampling <https://en.wikipedia.org/wiki/Latin_hypercube_sampling>`__
is an example of stochastic DOE algorithm:
given an input dimension and a number of samples,
running the algorithm twice will give two different DOEs.

For the sake of reproducibility,
|g| uses a `random seed <https://en.wikipedia.org/wiki/Random_seed>`__:
given an input dimension, a number of samples and a random seed,
running the algorithm twice will give the same DOE.

In this example,
we will see how |g| uses the random seed
and how the user can change its value.
"""

from __future__ import annotations

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import execute_algo
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction

# %%
# At the scenario level
# ---------------------
# First,
# we illustrate the use of the random seed at the :class:`.DOEScenario` level
# which is the appropriate level for most users.
# Then,
# we will illustrate this use at the :class:`.OptimizationProblem` level
# which can be useful for developers.
#
# Let us consider an :class:`.MDODiscipline` representing the function :math:`y=x^2`:
discipline = create_discipline("AnalyticDiscipline", expressions={"y": "x**2"})

# %%
# This function is defined over the interval :math:`[-1,1]`:
design_space = create_design_space()
design_space.add_variable("x", l_b=-1, u_b=1)

# %%
# We want to sample this discipline over this design space.
# For that,
# we express the sampling problem as a :class:`.DOEScenario`:
scenario = create_scenario(
    [discipline], "DisciplinaryOpt", "y", design_space, scenario_type="DOE"
)

# %%
# and solve it:
scenario.execute({"algo": "OT_OPT_LHS", "n_samples": 2})
scenario.formulation.opt_problem.database.get_last_n_x_vect(2)

# %%
# You can get the value of the random seed that has been used:
scenario._lib.seed

# %%
# When using the same DOE algorithm,
# a new call to :meth:`.DOEScenario.execute` increments the :attr:`.DOELibrary.seed`.
# Then,
# solving again this problem with the same configuration leads to a new result:
scenario.execute({"algo": "OT_OPT_LHS", "n_samples": 2})
scenario.formulation.opt_problem.database.get_last_n_x_vect(2)

# %%
# and we can check that the value of the seed was incremented:
scenario._lib.seed

# %%
# You can also pass a custom ``"seed"`` to the DOE algorithm
# with the key ``"algo_options"`` of the ``input_data``
# passed to :meth:`.DOEScenario.execute`:
scenario.execute({"algo": "OT_OPT_LHS", "n_samples": 2, "algo_options": {"seed": 123}})
scenario.formulation.opt_problem.database.get_last_n_x_vect(2)

# %%
# You can verify that :attr:`.DOELibrary.seed` has not been replaced by the custom value
# but incremented:
scenario._lib.seed

# %%
# At the problem level
# --------------------
# Basic
# ~~~~~
# Let us consider an :class:`.MDOFunction` representing the function :math:`y=x^2`:
function = MDOFunction(lambda x: x**2, "f", input_names=["x"], output_names=["y"])

# %%
# and defined over the unit interval :math:`x\in[0,1]`:
design_space = create_design_space()
design_space.add_variable("x", l_b=-1, u_b=1)

# %%
# We want to sample this function over this design space.
# For that,
# we express the sampling problem as an :class:`.OptimizationProblem`:
problem = OptimizationProblem(design_space)
problem.objective = function

# %%
# and solve it:
execute_algo(problem, "OT_OPT_LHS", algo_type="doe", n_samples=2)
problem.database.get_last_n_x_vect(2)

# %%
# Note:
#     We use the function :func:`.execute_algo`
#     as the :class:`.OptimizationProblem` does not have a method :func:`execute`
#     unlike the :class:`.Scenario`.
#
# Solving again this problem with the same configuration leads to the same result:
execute_algo(problem, "OT_OPT_LHS", algo_type="doe", n_samples=2)
problem.database.get_last_n_x_vect(2)

# %%
# and the result is still the same if we take 1 as random seed,
# as 1 is the default value of this seed:
execute_algo(problem, "OT_OPT_LHS", algo_type="doe", n_samples=2, seed=1)
problem.database.get_last_n_x_vect(2)

# %%
# If you want to use a different random seed,
# you only have to change the value of ``seed``:
execute_algo(problem, "OT_OPT_LHS", algo_type="doe", n_samples=2, seed=3)
problem.database.get_last_n_x_vect(2)

# %%
# Advanced
# ~~~~~~~~
# You can also solve your problem with a lower level API
# by directly instantiating the :class:`.DOELibrary` of interest.
# A :class:`.DOELibrary` has a :attr:`~.DOELibrary.seed` initialized at 0
# that is incremented at the beginning of each execution:
library = OpenTURNS()
library.algo_name = "OT_OPT_LHS"
library.execute(problem, n_samples=2)
library.seed, problem.database.get_last_n_x_vect(2)
# %%
# Solving again the problem will give different samples:
library.execute(problem, n_samples=2)
library.seed, problem.database.get_last_n_x_vect(2)
# %%
# You can also change the seed:
library.seed = 123
library.execute(problem, n_samples=2)
library.seed, problem.database.get_last_n_x_vect(2)
