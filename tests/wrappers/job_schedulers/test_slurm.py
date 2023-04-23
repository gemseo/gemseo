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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path
from subprocess import CalledProcessError

import pytest
from gemseo import create_discipline
from gemseo.wrappers.job_schedulers.slurm import SLURM
from pytest import fixture


@fixture
def discipline(tmpdir):
    """Create a JobSchedulerDisciplineWrapper based on JobSchedulerDisciplineWrapper
    using the SLURM template.

    Returns:
        The wrapped discipline.
    """
    disc = SLURM(
        discipline=create_discipline("SobieskiMission"),
        workdir_path=tmpdir,
        scheduler_run_command="python",
        job_template_path=Path(__file__).parent / "mock_job_scheduler.py",
        job_out_filename="run_disc.py",
    )

    return disc


def test_run(discipline):
    """Tests the outputs written by the discipline."""
    assert "y_4" in discipline.execute()


def test_wrap_discipline_in_job_scheduler(tmpdir):
    """Test the LSF wrapper execution errors when LSF is not available."""
    disc = create_discipline("SobieskiMission")
    wrapped = SLURM(disc, workdir_path=tmpdir)

    with pytest.raises(
        CalledProcessError, match="sbatch .* returned non-zero exit status 1."
    ):
        wrapped.execute()
