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
from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import pytest

from gemseo import create_discipline
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.deserialize_and_run import _parse_inputs
from gemseo.utils.deserialize_and_run import _run_discipline_save_outputs
from gemseo.utils.deserialize_and_run import main
from gemseo.utils.path_discipline import PathDiscipline


@pytest.fixture()
def discipline_and_data(tmpdir):
    tmpdir = Path(tmpdir)
    path_to_discipline = tmpdir / "discipline.pckl"
    discipline = create_discipline("SobieskiMission")
    discipline.to_pickle(path_to_discipline)
    path_to_outputs = tmpdir / "outputs.pckl"
    path_to_input_data = tmpdir / "inputs.pckl"
    with open(path_to_input_data, "wb") as outf:
        pickler = pickle.Pickler(outf, protocol=2)
        pickler.dump(discipline.default_inputs)
    return path_to_discipline, path_to_outputs, path_to_input_data, discipline


@pytest.fixture()
def sys_argv(discipline_and_data):
    (
        path_to_discipline,
        path_to_outputs,
        path_to_input_data,
        _,
    ) = discipline_and_data
    tmpdir = path_to_discipline.parent
    return [
        str(tmpdir),
        str(path_to_discipline),
        str(path_to_input_data),
        str(path_to_outputs),
    ]


def test_parse_inputs(discipline_and_data, sys_argv):
    """Test the input parsing for the deserialize_and_run executable."""
    (
        path_to_discipline,
        path_to_outputs,
        path_to_input_data,
        _,
    ) = discipline_and_data
    tmpdir = path_to_discipline.parent
    workir_path, serialized_disc_path, input_data_path, outputs_path = _parse_inputs(
        sys_argv
    )
    assert Path(workir_path) == tmpdir
    assert Path(serialized_disc_path) == path_to_discipline
    assert Path(input_data_path) == path_to_input_data
    assert Path(outputs_path) == path_to_outputs


def test_parse_inputs_fail(tmpdir):
    """Test the input parsing failure handling."""
    with pytest.raises(SystemExit):
        _parse_inputs(["1"])

    idontexist_path = str(tmpdir / "idontexist")
    i_exit_path = Path(tmpdir) / "dummy.txt"
    i_exit_path.write_text("a", encoding="utf8")
    i_exist = str(i_exit_path)

    with pytest.raises(FileNotFoundError, match="Work directory.*does not exist."):
        _parse_inputs([idontexist_path, i_exist, i_exist, i_exist])

    with pytest.raises(SystemExit):
        _parse_inputs([i_exist, idontexist_path, i_exist, i_exist])

    with pytest.raises(SystemExit):
        _parse_inputs([i_exist, i_exist, idontexist_path, i_exist])

    _parse_inputs([i_exist, i_exist, i_exist, i_exist])


def test_run_discipline_save_outputs(discipline_and_data):
    """Test the run and save outputs."""
    (
        path_to_discipline,
        _,
        _,
        discipline,
    ) = discipline_and_data
    workir_path = path_to_discipline.parent
    outputs_path = workir_path / "outputs.pckl"
    _run_discipline_save_outputs(
        discipline, discipline.default_inputs, outputs_path, workir_path
    )
    assert Path(outputs_path).exists()
    with open(outputs_path, "rb") as infile:
        outputs = pickle.load(infile)
    assert compare_dict_of_arrays(outputs, discipline.execute())


def test_run_discipline_save_outputs_errors(discipline_and_data):
    """Test the outputs saving error handling."""
    error_message = "I failed"

    def _run_and_fail():
        raise ValueError(error_message)

    (
        path_to_discipline,
        _,
        _,
        discipline,
    ) = discipline_and_data
    discipline._run = _run_and_fail

    workir_path = path_to_discipline.parent
    outputs_path = workir_path / "outputs.pckl"
    return_code = _run_discipline_save_outputs(
        discipline, discipline.default_inputs, outputs_path, workir_path
    )
    assert return_code == 1
    with outputs_path.open("rb") as discipline_file:
        error, _ = pickle.load(discipline_file)
        assert isinstance(error, ValueError)
        assert error.args[0] == error_message


def test_main():
    """Test the main entry point."""
    with pytest.raises(SystemExit):
        main()


def test_path_serialization(tmp_path):
    """Test the execution of a serialized discipline that contains Paths."""

    path_to_discipline = tmp_path / "discipline.pckl"
    discipline = PathDiscipline(tmp_path)
    discipline.to_pickle(path_to_discipline)
    path_to_outputs = tmp_path / "outputs.pckl"
    path_to_input_data = tmp_path / "inputs.pckl"

    with open(path_to_input_data, "wb") as outf:
        pickler = pickle.Pickler(outf, protocol=2)
        pickler.dump(discipline.default_inputs)

    completed = subprocess.run(
        f"gemseo-deserialize-run {tmp_path} {path_to_discipline} "
        f"{path_to_input_data} {path_to_outputs}",
        shell=True,
        capture_output=True,
        cwd=tmp_path,
    )

    assert completed.returncode == 0

    out = discipline.execute()
    assert out["y"] == 1
