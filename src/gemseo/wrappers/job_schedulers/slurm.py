# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""SLURM interface."""
from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any

from gemseo.core.discipline import MDODiscipline
from gemseo.wrappers.job_schedulers.scheduler_wrapped_disc import (
    JobSchedulerDisciplineWrapper,
)

LOGGER = getLogger(__name__)


class SLURM(JobSchedulerDisciplineWrapper):
    """A discipline that wraps the execution of the SLURM Job scheduler.

    The discipline is serialized to the disk, its input too, then a job file is created
    from a template to execute it with the provided options. The submission command is
    launched, it will setup the environment, deserialize the discipline and its inputs,
    execute it and serialize the outputs. Finally, the deserialized outputs are returned
    by the wrapper.
    """

    def __init__(
        self,
        discipline: MDODiscipline,
        workdir_path: Path,
        scheduler_run_command: str = "sbatch --wait",
        job_out_filename: str = "batch.sh",
        job_template_path: Path | str | None = None,
        use_template=True,
        setup_cmd: str = "",
        user_email: str = "",
        wall_time: str = "24:00:00",
        ntasks: int = 1,
        cpus_per_task: int = 1,
        nodes_number: int = 1,
        ntasks_per_node: int = 1,
        mem_per_cpu: str = "1G",
        **options: dict[str:Any],
    ) -> None:
        """
        Args:
            user_email: The user email to send the run status.
            wall_time: The wall time.
            ntasks: The number of tasks.
            cpus_per_task: The number of CPUS per task.
            nodes_number:  The number nodes.
            ntasks_per_node: The number of tasks per node.
            mem_per_cpu: The memory per CPU.
        """  # noqa:D205 D212 D415
        super().__init__(
            discipline=discipline,
            workdir_path=workdir_path,
            scheduler_run_command=scheduler_run_command,
            job_out_filename=job_out_filename,
            job_template_path=job_template_path,
            use_template=use_template,
            setup_cmd=setup_cmd,
            user_email=user_email,
            wall_time=wall_time,
            ntasks=ntasks,
            ntasks_per_node=ntasks_per_node,
            cpus_per_task=cpus_per_task,
            nodes_number=nodes_number,
            mem_per_cpu=mem_per_cpu,
            **options,
        )
