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
Parameter space
===============

In this example, we will see the basics of :class:`.ParameterSpace`.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix

configure_logger()


# %%
# Firstly, a :class:`.ParameterSpace` does not require any mandatory argument.
#
# Create a parameter space
# ------------------------
parameter_space = ParameterSpace()

# %%
# Then, we can add either deterministic variables
# from their lower and upper bounds
# (use :meth:`.ParameterSpace.add_variable`):
parameter_space.add_variable("x", l_b=-2.0, u_b=2.0)

# %%
# or uncertain variables from their distribution names and parameters
# (use :meth:`.ParameterSpace.add_random_variable`):
parameter_space.add_random_variable("y", "SPNormalDistribution", mu=0.0, sigma=1.0)
parameter_space

# %%
# .. warning::
#
#    We cannot mix probability distributions from different families,
#    e.g. an :class:`.OTDistribution` and a :class:`.SPDistribution`.
#
# We can check that the variables *x* and *y* are implemented
# as deterministic and uncertain variables respectively:
parameter_space.is_deterministic("x"), parameter_space.is_uncertain("y")

# %%
# Note that when GEMSEO does not offer a class for the SciPy distribution,
# we can use the generic GEMSEO class :class:`.SPDistribution`
# to create any SciPy distribution
# by setting ``interfaced_distribution`` to its SciPy name
# and ``parameters`` as a dictionary of SciPy parameter names and values
# (`see the documentation of SciPy
# <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html>`__).

# parameter_space.add_random_variable(
#     "y",
#     "SPDistribution",
#     interfaced_distribution="norm",
#     parameters={"loc": 1.0, "scale": 2.0},
# )

# %%
# A similar procedure can be followed
# for OpenTURNS distributions for which
# GEMSEO does not offer a class directly.
# We can use the generic GEMSEO class :class:`.OTDistribution`
# to create any OpenTURNS distribution
# by setting ``interfaced_distribution`` to its OpenTURNS name
# and ``parameters`` as a tuple of OpenTURNS parameter values
# (`see the documentation of OpenTURNS
# <https://openturns.github.io/openturns/latest/user_manual/_generated/
# openturns.Normal.html#openturns.Normal>`__).

# parameter_space.add_random_variable(
#     "y",
#     "OTDistribution",
#     interfaced_distribution="Normal",
#     parameters=(1.0, 2.0),
# )

# %%
# Sample from the parameter space
# -------------------------------
# We can sample the uncertain variables from the :class:`.ParameterSpace` and
# get values either as an array (default value):
sample = parameter_space.compute_samples(n_samples=2, as_dict=True)
sample

# %%
# or as a dictionary:
sample = parameter_space.compute_samples(n_samples=4)
sample

# %%
# Sample a discipline over the parameter space
# --------------------------------------------
# We can also sample a discipline over the parameter space. For simplicity,
# we instantiate an :class:`.AnalyticDiscipline` from a dictionary of
# expressions.
discipline = create_discipline("AnalyticDiscipline", expressions={"z": "x+y"})

# %%
# From these parameter space and discipline, we build a :class:`.DOEScenario`
# and execute it with a Latin Hypercube Sampling algorithm and 100 samples.
#
# .. warning::
#
#    A :class:`.Scenario` deals with all variables available in the
#    :class:`.DesignSpace`. By inheritance, a :class:`.DOEScenario` deals
#    with all variables available in the :class:`.ParameterSpace`.
#    Thus, if we do not filter the uncertain variables, the
#    :class:`.DOEScenario` will consider all variables. In particular, the
#    deterministic variables will be considered as uniformly distributed.
scenario = create_scenario(
    [discipline], "DisciplinaryOpt", "z", parameter_space, scenario_type="DOE"
)
scenario.execute({"algo": "lhs", "n_samples": 100})

# %%
# We can visualize the result by encapsulating the database in
# a :class:`.Dataset`:
dataset = scenario.to_dataset(opt_naming=False)

# %%
# This visualization can be tabular for example:
dataset

# %%
# or graphical by means of a scatter plot matrix for example:
ScatterMatrix(dataset).execute(save=False, show=True)

# %%
# Sample a discipline over the uncertain space
# --------------------------------------------
# If we want to sample a discipline over the uncertain space,
# we need to filter the uncertain variables:
parameter_space.filter(parameter_space.uncertain_variables)

# %%
# Then, we create a new scenario from this parameter space
# containing only the uncertain variables and execute it.
scenario = create_scenario(
    [discipline], "DisciplinaryOpt", "z", parameter_space, scenario_type="DOE"
)
scenario.execute({"algo": "lhs", "n_samples": 100})

# %%
# Finally, we build a dataset from the disciplinary cache and visualize it.
# We can see that the deterministic variable 'x' is set to its default
# value for all evaluations, contrary to the previous case where we were
# considering the whole parameter space.
dataset = scenario.to_dataset(opt_naming=False)
dataset
