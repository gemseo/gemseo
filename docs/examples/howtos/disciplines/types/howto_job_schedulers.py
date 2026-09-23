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

"""# Interface with HPC job schedulers (SLURM, LSF, PBS, etc)

## Problem

You want to send any discipline,
sub-process such as an MDA,
or scenario
to an HPC using the job scheduler interfaces.

## Solution

The method to be used is
[wrap_discipline_in_job_scheduler][gemseo.wrap_discipline_in_job_scheduler]
to wrap any discipline
and [wrap_scenario_in_job_scheduler][gemseo.scenario.job_scheduler.wrap_scenario_in_job_scheduler]
to wrap a scenario.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo import wrap_discipline_in_job_scheduler
from gemseo.formulation.mdf_settings import MDF_Settings
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.scenario import MDOScenario
from gemseo.scenario import wrap_scenario_in_job_scheduler

# %%
# ### Prerequisites
#
# This how-to needs a discipline.

discipline = create_discipline("Sellar1")

# %%
# ### 1. Wrap your discipline
#
# The discipline is executed on 24 CPUs using the SLURM wrapper, on an HPC.

discipline_in_queue = wrap_discipline_in_job_scheduler(
    discipline,
    workdir_path="workdir",
    cpus_per_task=24,
    scheduler_name="SLURM",
)

# %%
# ### 2. Create a scenario
#
# An entire [EvaluationScenario][gemseo.scenario.evaluation.EvaluationScenario],
# e.g. an [MDOScenario][gemseo.scenario.mdo.MDOScenario],
# can also be sent to an HPC,
# i.e. the whole scenario execution rather than a single discipline evaluation.
# The algorithm must be set on the scenario before wrapping it,
# and the objective too when the scenario solves an optimization problem.

disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
design_space = SellarDesignSpace(add_couplings=False)
scenario = MDOScenario(disciplines, design_space, formulation_settings=MDF_Settings())
scenario.add_objective("obj")
scenario.add_constraint("c_1", constraint_type="ineq")
scenario.add_constraint("c_2", constraint_type="ineq")
scenario.set_algorithm(SLSQP_Settings(max_iter=20))

# %%
# ### 3. Wrap your scenario
#
# The scenario is adapted into a discipline using
# [MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter]
# (or [EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter]
# when the scenario does not solve an optimization problem)
# and wrapped in a job scheduler.
# As the scenario is driven by an optimization algorithm,
# the inputs of the wrapper are the design variables,
# whose values are used as the starting point of the optimization.
# A scenario driven by a DOE algorithm, which ignores the starting point,
# has no input by default.

scenario_in_queue = wrap_scenario_in_job_scheduler(
    scenario,
    workdir_path="workdir",
    cpus_per_task=24,
    scheduler_name="SLURM",
    adapter_settings={"save_databases": True},
)

# %%
# The execution of the wrapper returns
# the objective, the constraints, the observables and the design variables
# at the end of the scenario execution, and nothing else.
# The database and the result of the scenario remain in the remote process,
# so the local scenario cannot be post-processed.
# Consequently:
#
# - any other discipline output must be declared as an observable to be returned;
#   alternatively, `output_names` can be passed in `adapter_settings`,
#   but it replaces the default outputs,
#   so the objective, the constraints, the observables and the design variables
#   have to be repeated in it when they are still wanted,
# - for a scenario solving an evaluation problem and driven by a DOE,
#   the returned values are those of the last evaluated sample,
#   while an optimization scenario driven by a DOE returns those of the best sample,
# - the setting `save_databases` of `adapter_settings`
#   saves the database to an HDF5 file in the job working directory,
#   which can be imported afterwards.

# %%
# ## Summary
#
# Use the [wrap_discipline_in_job_scheduler][gemseo.wrap_discipline_in_job_scheduler]
# function to send any discipline to an HPC using a job scheduler
# and [wrap_scenario_in_job_scheduler][gemseo.scenario.job_scheduler.wrap_scenario_in_job_scheduler]
# to send a scenario.
