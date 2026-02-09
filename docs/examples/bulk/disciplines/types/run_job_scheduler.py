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
"""# Execute a discipline with job schedulers.

This example execute a DOE of 100 points on an MDA,
each MDA is executed on 24 CPUS using the SLURM wrapper, on a HPC,
and at most 10 points run in parallel,
everytime a point of the DOE is computed,
another one is submitted to the queue.
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo import create_mda
from gemseo import create_scenario
from gemseo import wrap_discipline_in_job_scheduler
from gemseo.core.discipline.discipline import Discipline
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.settings.doe import LHS_Settings

disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
mda = create_mda("MDAGaussSeidel", disciplines)
wrapped_mda = wrap_discipline_in_job_scheduler(
    mda,
    scheduler_name="SLURM",
    workdir_path="workdir",
    cpus_per_task=24,
)
scenario = create_scenario(
    mda,
    "obj",
    SellarDesignSpace(),
    formulation_name="DisciplinaryOpt",
    scenario_type="DOE",
)
scenario.execute(algo_name="LHS", n_samples=100, n_processes=10)

# %%
# In this variant,
# each discipline is wrapped independently in the job scheduler,
# which allows to parallelize more the process
# because each discipline will run on indpendent nodes,
# whithout being parallelized using MPI.
# The drawback is that each discipline execution will be queued on the HPC.
# A HDF5 cache is attached to the MDA,
# so all executions will be recorded.
# Each wrapped discipline can also be cached using a HDF cache.

disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
wrapped_disciplines = [
    wrap_discipline_in_job_scheduler(
        discipline,
        workdir_path="workdir",
        cpus_per_task=24,
        scheduler_name="SLURM",
    )
    for discipline in disciplines
]
scenario = create_scenario(
    wrapped_disciplines,
    "obj",
    SellarDesignSpace(),
    formulation_name="MDF",
    scenario_type="DOE",
)
scenario.formulation.mda.set_cache(
    Discipline.CacheType.HDF5, hdf_file_path="mda_cache.h5"
)
scenario.execute(LHS_Settings(n_samples=100, n_processes=10))
