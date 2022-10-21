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
Dataset from an optimization problem
====================================

In this example, we will see how to build a :class:`.Dataset` from objects
of an :class:`.OptimizationProblem`.
For that, we need to import this :class:`.Dataset` class:
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace

configure_logger()


##############################################################################
# Synthetic data
# --------------
# We can sample the :class:`.Sellar1` discipline and use the
# corresponding :class:`.OptimizationProblem`:

discipline = create_discipline("Sellar1")
design_space = SellarDesignSpace().filter(discipline.get_input_data_names())

scenario = create_scenario(
    [discipline], "DisciplinaryOpt", "y_1", design_space, scenario_type="DOE"
)
scenario.execute({"algo": "lhs", "n_samples": 5})
opt_problem = scenario.formulation.opt_problem

##############################################################################
# Create a dataset
# ----------------
# We can easily build a dataset from this :class:`.OptimizationProblem`:
# either by separating the design parameters from the function
# (default option):
dataset = opt_problem.export_to_dataset("sellar1_doe")
print(dataset)
##############################################################################
# or by considering all features as default parameters:
dataset = opt_problem.export_to_dataset("sellar1_doe", categorize=False)
print(dataset)
##############################################################################
# or by using an input-output naming rather than an optimization naming:
dataset = opt_problem.export_to_dataset("sellar1_doe", opt_naming=False)
print(dataset)
##############################################################################
# .. note::
#     Only design variables and functions (objective function, constraints) are
#     stored in the database. If you want to store state variables, you must add
#     them as observables before the problem is executed. Use the
#     :meth:`~gemseo.core.scenario.Scenario.add_observable` method.

##############################################################################
# Access properties
# -----------------
dataset = opt_problem.export_to_dataset("sellar1_doe")
##############################################################################
# Variables names
# ~~~~~~~~~~~~~~~
# We can access the variables names:
print(dataset.variables)

##############################################################################
# Variables sizes
# ~~~~~~~~~~~~~~~
# We can access the variables sizes:
print(dataset.sizes)

##############################################################################
# Variables groups
# ~~~~~~~~~~~~~~~~
# We can access the variables groups:
print(dataset.groups)

##############################################################################
# Access data
# -----------
# Access by group
# ~~~~~~~~~~~~~~~
# We can get the data by group, either as an array (default option):
print(dataset.get_data_by_group("design_parameters"))
##############################################################################
# or as a dictionary indexed by the variables names:
print(dataset.get_data_by_group("design_parameters", True))

##############################################################################
# Access by variable name
# ~~~~~~~~~~~~~~~~~~~~~~~
# We can get the data by variables names,
# either as a dictionary indexed by the variables names (default option):
print(dataset.get_data_by_names(["x_shared", "y_2"]))
##############################################################################
# or as an array:
print(dataset.get_data_by_names(["x_shared", "y_2"], False))

##############################################################################
# Access all data
# ~~~~~~~~~~~~~~~
# We can get all the data, either as a large array:
print(dataset.get_all_data())
##############################################################################
# or as a dictionary indexed by variables names:
print(dataset.get_all_data(as_dict=True))
##############################################################################
# We can get these data sorted by category, either with a large array for each
# category:
print(dataset.get_all_data(by_group=False))
##############################################################################
# or with a dictionary of variables names:
print(dataset.get_all_data(by_group=False, as_dict=True))
