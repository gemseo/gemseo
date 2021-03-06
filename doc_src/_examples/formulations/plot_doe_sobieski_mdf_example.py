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
#        :author: François Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
MDF-based DOE on the Sobieski SSBJ test case
============================================
"""
from os import name as os_name

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from matplotlib import pyplot as plt

configure_logger()

##############################################################################
# Instantiate the  disciplines
# ----------------------------
# First, we instantiate the four disciplines of the use case:
# :class:`~gemseo.problems.sobieski.disciplines.SobieskiPropulsion`,
# :class:`~gemseo.problems.sobieski.disciplines.SobieskiAerodynamics`,
# :class:`~gemseo.problems.sobieski.disciplines.SobieskiMission`
# and :class:`~gemseo.problems.sobieski.disciplines.SobieskiStructure`.
disciplines = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
        "SobieskiStructure",
    ]
)

##############################################################################
# Build, execute and post-process the scenario
# --------------------------------------------
# Then, we build the scenario which links the disciplines
# with the formulation and the optimization algorithm. Here, we use the
# :class:`.BiLevel` formulation. We tell the scenario to minimize -y_4
# instead of minimizing y_4 (range), which is the default option.
#
# We need to define the design space.
design_space = SobieskiProblem().design_space

##############################################################################
# Instantiate the scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^
scenario = create_scenario(
    disciplines,
    formulation="MDF",
    objective_name="y_4",
    design_space=design_space,
    maximize_objective=True,
    scenario_type="DOE",
)

##############################################################################
# Set the design constraints
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, "ineq")

##############################################################################
# Execute the scenario
# ^^^^^^^^^^^^^^^^^^^^
# Use provided analytic derivatives
scenario.set_differentiation_method("user")

##############################################################################
# Multiprocessing
# ^^^^^^^^^^^^^^^
# It is possible to run a DOE in parallel using multiprocessing, in order to do
# this, we specify the number of processes to be used for the computation of
# the samples.

#    The multiprocessing option has some limitations on Windows.
#    Due to problems with sphinx, we disable it in this example.
#    For Python versions < 3.7 and Numpy < 1.20.0, subprocesses may get hung
#    randomly during execution. It is strongly recommended to update your
#    environment to avoid this problem.
#    The features :class:`.MemoryFullCache` and :class:`.HDF5Cache` are not
#    available for multiprocessing on Windows.
#    As an alternative, we recommend the method
#    :meth:`.DOEScenario.set_optimization_history_backup`.

##############################################################################
# We define the algorithm options. Here the criterion = center option of pyDOE
# centers the points within the sampling intervals.
algo_options = {
    "criterion": "center",
    # Evaluate gradient of the MDA
    # with coupled adjoint
    "eval_jac": True,
    # Run in parallel on 1 or 4 processors
    "n_processes": 1 if os_name == "nt" else 4,
}

scenario.execute({"n_samples": 30, "algo": "lhs", "algo_options": algo_options})

##############################################################################
# .. warning::
#    On Windows, the progress bar may show duplicated instances during the
#    initialization of each subprocess. In some cases it may also print the
#    conclusion of an iteration ahead of another one that was concluded first.
#    This is a consequence of the pickling process and does not affect the
#    computations of the scenario.

##############################################################################
# Plot the optimization history view
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process("OptHistoryView", show=False, save=False)

###############################################################################
# .. tip::
#
#    Each post-processing method requires different inputs and offers a variety
#    of customization options. Use the API function
#    :meth:`~gemseo.api.get_post_processing_options_schema` to print a table with
#    the attributes for any post-processing algo. Or refer to our dedicated page:
#    :ref:`gen_post_algos`.

##############################################################################
# Plot the scatter matrix
# ^^^^^^^^^^^^^^^^^^^^^^^
scenario.post_process(
    "ScatterPlotMatrix", show=False, save=False, variable_names=["y_4", "x_shared"]
)

##############################################################################
# Plot correlations
# ^^^^^^^^^^^^^^^^^
scenario.post_process("Correlations", show=False, save=False)
# Workaround for HTML rendering, instead of ``show=True``
plt.show()
