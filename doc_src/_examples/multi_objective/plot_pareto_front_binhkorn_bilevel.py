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
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Pareto front on the Binh and Korn problem using a BiLevel formulation
=====================================================================

In this example,
we illustrate the computation of a Pareto front plot for the Binh and Korn problem.
We use a BiLevel formulation in order to only compute the Pareto-optimal points.
"""

# %%
# Import
# ------
# The first step is to import some high-level functions
# and to configure the logger.
from __future__ import annotations

from copy import deepcopy
from logging import WARNING

from numpy import array

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import execute_post

configure_logger()


# %%
# Definition of the disciplines
# -----------------------------
#
# In this example,
# we create the Binh and Korn disciplines from scratch by declaring
# their expressions and using the :class:`.AnalyticDiscipline`.

expr_binh_korn = {
    "obj1": "4*x1**2 + 4*x2**2",
    "obj2": "(x1-5.)**2 + (x2-5.)**2",
    "cstr1": "(x1-5.)**2 + x2**2 - 25.",
    "cstr2": "-(x1-8.)**2 - (x2+3)**2 + 7.7",
}

# %%
# This constraint will be used to set `obj1` to a target value for
# the lower-level scenario.

expr_cstr_obj1_target = {"cstr3": "obj1 - obj1_target"}

# %%
# Instantiation of the disciplines
# --------------------------------
#
# Here, we create the disciplines from their expressions.

discipline_binh_korn = create_discipline(
    "AnalyticDiscipline", expressions=expr_binh_korn
)
discipline_cstr_obj1 = create_discipline(
    "AnalyticDiscipline", expressions=expr_cstr_obj1_target
)

# %%
# Definition of the lower-level design space
# ------------------------------------------

design_space = create_design_space()
design_space.add_variable(
    "x1", lower_bound=array([0.0]), upper_bound=array([5.0]), value=array([2.0])
)
design_space.add_variable(
    "x2", lower_bound=array([-5.0]), upper_bound=array([3.0]), value=array([2.0])
)

disciplines = [
    discipline_binh_korn,
    discipline_cstr_obj1,
]

# %%
# Creation of the lower-level scenario
# ------------------------------------
# This scenario aims at finding the `obj2` optimal value for a specific value of `obj1`.

sub_scenario = create_scenario(
    disciplines,
    "obj2",
    design_space,
    formulation_name="DisciplinaryOpt",
)

sub_scenario.set_algorithm(algo_name="NLOPT_SLSQP", max_iter=100)

# %%
# We add the Binh and Korn problem constraints.
sub_scenario.add_constraint("cstr1", constraint_type="ineq")
sub_scenario.add_constraint("cstr2", constraint_type="ineq")

# %%
# Creation of the design space for the system-level scenario
# ----------------------------------------------------------
# At the system level, we will fix a target for the `obj1` value of
# the lower-level scenario.
system_design_space = create_design_space()
system_design_space.add_variable(
    "obj1_target",
    lower_bound=array([0.1]),
    upper_bound=array([100.0]),
    value=array([1.0]),
)

# %%
# Creation of the system-level DOE Scenario
# -----------------------------------------
# The system-level scenario will perform a DOE over the `obj1_target` variable.
# We will use the `BiLevel` formulation to nest the lower-level scenario into the DOE.
# The log level for the sub scenarios is set to `WARNING` to avoid getting the complete
# log of each sub scenario, which would be too verbose. Set it to `INFO` if you wish to
# keep the logs of each sub scenario as well.

system_scenario = create_scenario(
    sub_scenario,
    "obj1",
    system_design_space,
    scenario_type="DOE",
    formulation_name="BiLevel",
    sub_scenarios_log_level=WARNING,
)

# %%
# .. tip::
#
#    When running BiLevel scenarios, it is interesting to access the optimization
#    history of the sub-scenarios for each system iteration. By default, the setting
#    ``keep_opt_history`` is set to ``True``. This allows you to store in memory the
#    databases of the sub-scenarios (see the last section of this example for more
#    details).
#    In some cases, storing the databases in memory can take up too much space and cause
#    performance issues. In these cases, set ``keep_opt_history=False`` and save the
#    databases to the disk using ``save_opt_history=True``. If your sub-scenarios are
#    running in parallel, and you are saving the optimization histories to the disk, set
#    the ``naming`` setting to ``"UUID"``, which is multiprocessing-safe.
#    The setting ``keep_opt_history`` will not work if the sub-scenarios are running in
#    parallel because the databases are not copied from the sub-processes to the main
#    process. In this case you shall always save the optimization history to the disk.

# %%
# Add the system-level constraint and observables
# -----------------------------------------------
# Here, we add the constraint on the `obj1_target`, this way we make sure that the
# lower-level scenario will respect the target imposed by the system.
# The BiLevel formulation will automatically add the constraints from the system-level
# to the lower-level, if you wish to handle the constraints manually, pass
# `apply_cstr_tosub_scenarios=False` as an argument to `create_scenario`.
# Note that `obj2` shall be added as an observable of `scenario_doe`,
# otherwise it cannot be used by the ParetoFront post-processing.
system_scenario.add_constraint("cstr3")
system_scenario.add_observable("obj2")
system_scenario.xdsmize(save_html=False, pdf_build=False)
# %%
# Run the scenario
# ----------------
# Finally, we run a full-factorial DOE using 100 samples and run the post-processing.
system_scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=50)
system_scenario.post_process(
    post_name="ParetoFront", objectives=["obj1", "obj2"], save=False, show=True
)

# %%
# Plot the sub-scenario histories of the 2 first system iterations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The code below will not work if you ran the system scenario with ``n_processes`` > 1.
# Indeed, parallel execution of sub-scenarios prevents us to save the databases from
# each sub-process to the main process. If you ran the system scenario with many
# processes, you can still save the databases to the disk with
# ``save_opt_history=True`` and ``naming="UUID"``. Refer to the formulation settings for
# more information.
sub_scenario_databases = system_scenario.formulation.scenario_adapters[0].databases
for database in sub_scenario_databases[:2]:
    opt_problem = deepcopy(sub_scenario.formulation.optimization_problem)
    opt_problem.database = database
    execute_post(opt_problem, post_name="OptHistoryView", save=False, show=True)
