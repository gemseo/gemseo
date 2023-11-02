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

import pickle
import re
from pathlib import Path
from string import Template

import pytest

from gemseo import create_discipline
from gemseo import wrap_discipline_in_job_scheduler
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.platform import PLATFORM_IS_WINDOWS
from gemseo.wrappers import job_schedulers
from gemseo.wrappers.job_schedulers.scheduler_wrapped_disc import (
    JobSchedulerDisciplineWrapper,
)


@pytest.fixture()
def discipline(tmpdir):
    """Create a JobSchedulerDisciplineWrapper based on JobSchedulerDisciplineWrapper
    using the SLURM template.

    Returns:
        The wrapped discipline.
    """
    template_path = Path(job_schedulers.__file__).parent / "templates" / "SLURM"
    return JobSchedulerDisciplineWrapper(
        discipline=create_discipline("SobieskiMission"),
        workdir_path=tmpdir,
        scheduler_run_command="sbatch",
        job_template_path=template_path,
        user_email="toto@irt.com",
        ntasks=1,
        ntasks_per_node=1,
        cpus_per_task=1,
        nodes_number=1,
        mem_per_cpu="2G",
        wall_time="1:0:0",
    )


@pytest.fixture()
def discipline_mocked_js(tmpdir) -> JobSchedulerDisciplineWrapper:
    """Creates a JobSchedulerDisciplineWrapper based on JobSchedulerDisciplineWrapper
    using the mock template.

    Returns:
        The wrapped discipline
    """
    return JobSchedulerDisciplineWrapper(
        create_discipline("SobieskiMission"),
        job_template_path=Path(__file__).parent / "mock_job_scheduler.py",
        workdir_path=tmpdir,
        job_out_filename="run_disc.py",
        scheduler_run_command="python",
    )


def test_write_inputs_to_disk(discipline, tmpdir):
    """Test the outputs written by the discipline."""
    current_workdir = Path(tmpdir)
    path_to_discipline, path_to_input_data = discipline._write_inputs_to_disk(
        current_workdir
    )
    assert path_to_discipline.exists()
    assert path_to_discipline.parent == current_workdir
    assert path_to_input_data.exists()
    assert path_to_input_data.parent == current_workdir


def test_generate_job_template(discipline):
    """Test the job scheduler template creation."""
    current_workdir = discipline._create_current_workdir()
    path_to_discipline, path_to_input_data = discipline._write_inputs_to_disk(
        current_workdir
    )
    path_to_outputs = current_workdir / "outputs.pckl"
    log_file_path = current_workdir / "logging.log"
    dest_job_file_path = discipline._generate_job_file_from_template(
        current_workdir,
        path_to_discipline,
        path_to_input_data,
        path_to_outputs,
        log_file_path,
    )
    assert dest_job_file_path.exists()
    with open(dest_job_file_path) as infile:
        lines = infile.readlines()
    for line in lines:
        if "#SBATCH" in line:
            assert "$" not in line
    assert len(lines) > 40


def test_generate_job_template_fail(discipline, tmpdir):
    """Test that missing template values raises a proper exception."""
    discipline.job_file_template = Template("$missing")
    with pytest.raises(
        KeyError, match="Value not passed to template for key: 'missing'"
    ):
        discipline._generate_job_file_from_template(tmpdir, None, None, None, None)


def test_run_fail(discipline: JobSchedulerDisciplineWrapper, tmpdir, caplog):
    """Test the run failure is correctly handled."""
    discipline._scheduler_run_command = "IDONTEXIST"
    if PLATFORM_IS_WINDOWS:
        match = r"\[WinError 2\] .*"
    else:
        match = re.escape("[Errno 2] No such file or directory: 'IDONTEXIST'")
    with pytest.raises(FileNotFoundError, match=match):
        discipline._run_command(tmpdir, tmpdir / "output.pckl")


def test_handle_outputs_errors(discipline: JobSchedulerDisciplineWrapper, tmpdir):
    """Test that errors in outputs are correctly handled."""
    with pytest.raises(
        FileNotFoundError,
        match="The serialized outputs file of the discipline does not exist",
    ):
        discipline._handle_outputs(Path("IDONTEXIST"), tmpdir)

    exception = (ValueError("An error."), "stack trace.")
    outputs_path = tmpdir / "outputs.pickl"
    with open(outputs_path, "wb") as outf:
        outf.write(pickle.dumps(exception))

    with pytest.raises(ValueError, match="An error."):
        discipline._handle_outputs(outputs_path, tmpdir)


def test_create_current_workdir(discipline):
    """Test the creation of the workdir."""
    current_workdir = discipline._create_current_workdir()
    assert current_workdir.exists()
    assert current_workdir.parent == discipline._workdir_path


def test_execution(discipline_mocked_js):
    """Test the execution of the wrapped discipline."""
    orig_disc = discipline_mocked_js._discipline
    ref_data = orig_disc.default_inputs
    ref_data["x_shared"] += 1.0
    out = discipline_mocked_js.execute(ref_data)
    assert "y_4" in out
    mission_local = create_discipline("SobieskiMission")
    out_ref = mission_local.execute(ref_data)
    assert compare_dict_of_arrays(out, out_ref)


def test_api_fail(tmpdir):
    """Test the api method that wraps the JS error messages."""
    with pytest.raises(
        FileNotFoundError,
        match=r"Job scheduler template file .*IDONTEXIST does not exist.",
    ):
        wrap_discipline_in_job_scheduler(
            create_discipline("SobieskiMission"),
            "SLURM",
            job_template_path="IDONTEXIST",
            workdir_path=tmpdir,
            job_out_filename="run_disc.py",
            scheduler_run_command="python",
        )
