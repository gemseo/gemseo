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
Scalable problem
================

We want to solve the Aerostructure MDO problem
by means of the :class:`.MDF` formulation
with a higher dimension for the sweep parameter.
For that, we use the :class:`.ScalableProblem` class.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.aerostructure.aerostructure_design_space import (
    AerostructureDesignSpace,
)
from gemseo.problems.scalable.data_driven.problem import ScalableProblem

configure_logger()


###############################################################################
# Define the design problem
# -------------------------
# In a first step, we define the design problem in terms of
# objective function (to maximize or minimize),
# design variables (local and global)
# and constraints (equality and inequality).
design_variables = ["thick_airfoils", "thick_panels", "sweep"]
objective_function = "range"
eq_constraints = ["c_rf"]
ineq_constraints = ["c_lift"]
maximize_objective = True

###############################################################################
# Create the disciplinary datasets
# --------------------------------
# Then, we create the disciplinary :class:`.AbstractFullCache` datasets
# based on a :class:`.DiagonalDOE`.
disciplines = create_discipline(["Aerodynamics", "Structure", "Mission"])
datasets = []
for discipline in disciplines:
    design_space = AerostructureDesignSpace()
    design_space.filter(discipline.get_input_data_names())
    output_names = iter(discipline.get_output_data_names())
    scenario = create_scenario(
        discipline,
        "DisciplinaryOpt",
        next(output_names),
        design_space,
        scenario_type="DOE",
    )
    for output_name in output_names:
        scenario.add_observable(output_name)
    scenario.execute({"algo": "DiagonalDOE", "n_samples": 10})
    datasets.append(scenario.export_to_dataset(name=discipline.name, opt_naming=False))

###############################################################################
# Instantiate a scalable problem
# ------------------------------
# In a third stage, we instantiate a :class:`.ScalableProblem`
# from these disciplinary datasets and from the definition of the MDO problem.
# We also increase the dimension of the sweep parameter.
problem = ScalableProblem(
    datasets,
    design_variables,
    objective_function,
    eq_constraints,
    ineq_constraints,
    maximize_objective,
    sizes={"sweep": 2},
)
print(problem)

###############################################################################
# .. note::
#
#    We could also provide options to the :class:`.ScalableModel` objects
#    by means of the constructor of :class:`.ScalableProblem`,
#    e.g. :code:`fill_factor` in the frame of the :class:`.ScalableDiagonalModel`.
#    In this example, we use the standard ones.

###############################################################################
# Visualize the N2 chart
# ----------------------
# We can see the coupling between disciplines through this N2 chart:
problem.plot_n2_chart(save=False, show=True)

###############################################################################
# Create an MDO scenario
# ----------------------
# Lastly, we create a :class:`.MDOScenario` with the :class:`.MDF` formulation
# and start the optimization at equilibrium,
# thus ensuring the feasibility of the first iterate.
scenario = problem.create_scenario("MDF", start_at_equilibrium=True)

###############################################################################
# .. note::
#
#    We could also provide options for the scalable models to the constructor
#    of :class:`.ScalableProblem`, e.g. :code:`fill_factor` in the frame of
#    the :class:`.ScalableDiagonalModel`.
#    In this example, we use the standard ones.

###############################################################################
# Once the scenario is created, we can execute it as any scenario.
# Here, we use the :code:`NLOPT_SLSQP` optimization algorithm
# with no more than 100 iterations.
scenario.execute({"algo": "NLOPT_SLSQP", "max_iter": 100})

###############################################################################
# We can post-process the results.
# Here, we use the standard :class:`.OptHistoryView`.
scenario.post_process("OptHistoryView", save=False, show=True)
