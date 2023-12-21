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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory to instantiate job scheduler interfaes from their class names."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.base_factory import BaseFactory
from gemseo.wrappers.job_schedulers.scheduler_wrapped_disc import (
    JobSchedulerDisciplineWrapper,
)

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.core.discipline import MDODiscipline


class SchedulersFactory(BaseFactory):
    """Job schedulers factory to create job scheduler interfaces from a name."""

    _CLASS = JobSchedulerDisciplineWrapper
    _MODULE_NAMES = ("gemseo.wrappers.job_schedulers",)

    def wrap_discipline(
        self,
        discipline: MDODiscipline,
        scheduler_name: str,
        workdir_path: Path,
        **options: dict[str, Any],
    ) -> JobSchedulerDisciplineWrapper:
        """Wrap the discipline within another one to delegate its execution to a job
        scheduler.

        The discipline is serialized to the disk, its input too, then a job file is
        created from a template to execute it with the provided options.
        The submission command is launched, it will setup the environment, deserialize
        the discipline and its inputs, execute it and serialize the outputs.
        Finally, the deserialized outputs are returned by the wrapper.

        All process classes :class:`~gemseo.core.mdo_scenario.MDOScenario`,
        or :class:`~gemseo.mda.mda.MDA`, inherit from
        :class:`~gemseo.core.discipline.MDODiscipline` so can be sent to HPCs this way.

        The job scheduler template script can be provided directly or the predefined
        templates file names in gemseo.wrappers.job_schedulers.template can be used.
        SLURM and LSF templates are provided, but one can use other job schedulers
        or to customize the scheduler commands according to the user needs
        and infrastructure requirements.

        The command to submit the job can also be overloaded.

        Args:
            discipline: The discipline to wrap in the job scheduler.
            scheduler_name: The name of the job scheduler
                (for instance LSF, SLURM, PBS).
            workdir_path: The path to the workdir
            **options: The submission options.

        Raises:
            OSError: If the job template does not exist.

        Warnings:
        This method serializes the passed discipline so it has to be serializable.
        All disciplines provided in |g| are serializable but it is possible that custom
        ones are not and this will make the submission process fail.
        """  # noqa:D205 D212 D415
        return self.create(
            scheduler_name,
            discipline=discipline,
            workdir_path=workdir_path,
            **options,
        )

    @property
    def scheduler_names(self) -> list[str]:
        """The names of the available job scheduler interfaces."""
        return self.class_names
