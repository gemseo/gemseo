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

import re
from pathlib import Path

import pytest

from gemseo import create_discipline
from gemseo.utils.platform import PLATFORM_IS_WINDOWS
from gemseo.wrappers.job_schedulers.slurm import SLURM


@pytest.fixture()
def discipline(tmpdir):
    """Create a JobSchedulerDisciplineWrapper based on JobSchedulerDisciplineWrapper
    using the SLURM template.

    Returns:
        The wrapped discipline.
    """
    return SLURM(
        discipline=create_discipline("SobieskiMission"),
        workdir_path=tmpdir,
        scheduler_run_command="python",
        job_template_path=Path(__file__).parent / "mock_job_scheduler.py",
        job_out_filename="run_disc.py",
    )


def test_run(discipline):
    """Tests the outputs written by the discipline."""
    assert "y_4" in discipline.execute()


def test_wrap_discipline_in_job_scheduler(tmpdir):
    """Test the LSF wrapper execution errors when LSF is not available."""
    disc = create_discipline("SobieskiMission")
    wrapped = SLURM(disc, workdir_path=tmpdir)

    if PLATFORM_IS_WINDOWS:
        match = r"\[WinError 2\] .*"
    else:
        match = re.escape("[Errno 2] No such file or directory: 'sbatch'")
    with pytest.raises(FileNotFoundError, match=match):
        wrapped.execute()
