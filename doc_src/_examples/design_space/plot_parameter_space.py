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

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario

configure_logger()


###############################################################################
# Firstly, a :class:`.ParameterSpace` does not require any mandatory argument.
#
# Create a parameter space
# ------------------------


parameter_space = ParameterSpace()


###############################################################################
# Then, we can add either deterministic variables from their lower and upper
# bounds (use :meth:`.DesignSpace.add_variable`)
# or uncertain variables from their distribution names and parameters
# (use :meth:`.add_random_variable`)

parameter_space.add_variable("x", l_b=-2.0, u_b=2.0)
parameter_space.add_random_variable("y", "SPNormalDistribution", mu=0.0, sigma=1.0)
print(parameter_space)

###############################################################################
# We can check that the deterministic and uncertain variables are implemented
# as deterministic and deterministic variables respectively:
print("x is deterministic: ", parameter_space.is_deterministic("x"))
print("y is deterministic: ", parameter_space.is_deterministic("y"))
print("x is uncertain: ", parameter_space.is_uncertain("x"))
print("y is uncertain: ", parameter_space.is_uncertain("y"))

###############################################################################
# Sample from the parameter space
# -------------------------------
# We can sample the uncertain variables from the :class:`.ParameterSpace` and
# get values either as an array (default value) or as a dictionary:
sample = parameter_space.compute_samples(n_samples=2, as_dict=True)
print(sample)
sample = parameter_space.compute_samples(n_samples=4)
print(sample)

###############################################################################
# Sample a discipline over the parameter space
# --------------------------------------------
# We can also sample a discipline over the parameter space. For simplicity,
# we instantiate an :class:`.AnalyticDiscipline` from a dictionary of
# expressions.

discipline = create_discipline("AnalyticDiscipline", expressions={"z": "x+y"})

###############################################################################
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
#    deterministic variables will be consider as uniformly distributed.

scenario = create_scenario(
    [discipline], "DisciplinaryOpt", "z", parameter_space, scenario_type="DOE"
)
scenario.execute({"algo": "lhs", "n_samples": 100})

###############################################################################
# We can visualize the result by encapsulating the database in
# a :class:`.Dataset`:
dataset = scenario.export_to_dataset(opt_naming=False)

###############################################################################
# This visualization can be tabular for example:
print(dataset.export_to_dataframe())

###############################################################################
# or graphical by means of a scatter plot matrix for example:
dataset.plot("ScatterMatrix")

###############################################################################
# Sample a discipline over the uncertain space
# --------------------------------------------
# If we want to sample a discipline over the uncertain space,
# we need to filter the uncertain variables:
parameter_space.filter(parameter_space.uncertain_variables)

###############################################################################
# Then, we create a new scenario from this parameter space
# containing only the uncertain variables and execute it.
scenario = create_scenario(
    [discipline], "DisciplinaryOpt", "z", parameter_space, scenario_type="DOE"
)
scenario.execute({"algo": "lhs", "n_samples": 100})

###############################################################################
# Finally, we build a dataset from the disciplinary cache and visualize it.
# We can see that the deterministic variable 'x' is set to its default
# value for all evaluations, contrary to the previous case where we were
# considering the whole parameter space.
dataset = scenario.export_to_dataset(opt_naming=False)
print(dataset.export_to_dataframe())
