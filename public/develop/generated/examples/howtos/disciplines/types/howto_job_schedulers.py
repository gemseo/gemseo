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

r"""# Interface with HPC job schedulers (SLURM, LSF, PBS, etc)

## Problem

How can I send any discipline,
or sub-process such as an MDA,
to an HPC using the job scheduler interfaces.

## Solution

The method to be used is
[wrap_discipline_in_job_scheduler][gemseo.wrap_discipline_in_job_scheduler]
to wrap any discipline.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo import wrap_discipline_in_job_scheduler

# %%
# ### 1. Create a discipline

discipline = create_discipline(["Sellar1"])


# ### 2. Wrap your discpline
#
# The discipline is executed on 24 CPUs using the SLURM wrapper, on an HPC.

discipline_in_queue = wrap_discipline_in_job_scheduler(
    discipline,
    workdir_path="workdir",
    cpus_per_task=24,
    scheduler_name="SLURM",
)

# %%
# ## Summary
#
# Use the [wrap_discipline_in_job_scheduler][gemseo.wrap_discipline_in_job_scheduler]
# function to send any discipline to an HPC using a job scheduler.
