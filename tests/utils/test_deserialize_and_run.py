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
from typing import TYPE_CHECKING

import pytest

from gemseo import create_discipline
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.deserialize_and_run import _parse_inputs
from gemseo.utils.deserialize_and_run import _run_discipline_save_outputs
from gemseo.utils.deserialize_and_run import main
from gemseo.utils.path_discipline import PathDiscipline
from gemseo.utils.pickle import to_pickle

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


@pytest.mark.parametrize("protocol", [pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL])
def discipline_and_data(protocol, tmp_wd):
    path_to_discipline = tmp_wd / "discipline.pckl"
    discipline = create_discipline("SobieskiMission")
    to_pickle(discipline, path_to_discipline)
    path_to_outputs = tmp_wd / "outputs.pckl"
    path_to_input_data = tmp_wd / "inputs.pckl"
    with path_to_input_data.open("wb") as outf:
        pickler = pickle.Pickler(outf, protocol=protocol)
        pickler.dump((discipline.io.input_grammar.defaults, (), ()))
    return path_to_discipline, path_to_outputs, path_to_input_data, discipline


@pytest.fixture
def sys_argv(discipline_and_data):
    (
        path_to_discipline,
        path_to_outputs,
        path_to_input_data,
        _,
    ) = discipline_and_data
    return [
        str(path_to_discipline),
        str(path_to_input_data),
        str(path_to_outputs),
    ]


@pytest.fixture
def test_parse_inputs(discipline_and_data, sys_argv) -> None:
    """Test the input parsing for the deserialize_and_run executable."""
    (
        path_to_discipline,
        path_to_outputs,
        path_to_input_data,
        _,
    ) = discipline_and_data
    (
        serialized_disc_path,
        input_data_path,
        outputs_path,
        linearize,
        execute_at_linearize,
    ) = _parse_inputs(sys_argv)
    assert serialized_disc_path == path_to_discipline
    assert input_data_path == path_to_input_data
    assert outputs_path == path_to_outputs
    assert not linearize
    assert not execute_at_linearize

    sys_argv.append("--linearize")
    outs = _parse_inputs(sys_argv)

    assert outs[-2]


@pytest.fixture(params=[pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL])
def test_parse_inputs_with_protocol(discipline_and_data, sys_argv, request) -> None:
    protocol = request.param
    """Test the input with a specific pickle protocol."""
    (
        path_to_discipline,
        path_to_outputs,
        path_to_input_data,
        _,
    ) = discipline_and_data
    sys_argv.append(f"--protocol={protocol}")
    (
        serialized_disc_path,
        input_data_path,
        outputs_path,
        linearize,
        execute_at_linearize,
        parsed_protocol,
    ) = _parse_inputs(sys_argv)
    assert serialized_disc_path == path_to_discipline
    assert input_data_path == path_to_input_data
    assert outputs_path == path_to_outputs
    assert not linearize
    assert not execute_at_linearize
    assert parsed_protocol == protocol


def test_parse_inputs_fail(tmp_wd) -> None:
    """Test the input parsing failure handling."""
    with pytest.raises(SystemExit):
        _parse_inputs(["1"])

    idontexist_path = str(tmp_wd / "idontexist")
    i_exit_path = tmp_wd / "dummy.txt"
    i_exit_path.write_text("a", encoding="utf8")
    i_exist = str(i_exit_path)

    with pytest.raises(SystemExit):
        _parse_inputs([idontexist_path, i_exist, i_exist])

    with pytest.raises(SystemExit):
        _parse_inputs([i_exist, idontexist_path, i_exist])

    _parse_inputs([i_exist, i_exist, i_exist])


@pytest.fixture
def test_run_discipline_save_outputs(discipline_and_data) -> None:
    """Test the run and save outputs."""
    (
        _path_to_discipline,
        _,
        _,
        discipline,
    ) = discipline_and_data
    outputs_path = Path("outputs.pckl")
    _run_discipline_save_outputs(
        discipline,
        discipline.io.input_grammar.defaults,
        outputs_path,
        False,
        False,
        (),
        (),
    )
    assert outputs_path.exists()
    with open(outputs_path, "rb") as infile:
        outputs, _ = pickle.load(infile)
    assert compare_dict_of_arrays(outputs, discipline.execute())


@pytest.fixture
def test_run_discipline_save_outputs_errors(discipline_and_data) -> None:
    """Test the outputs saving error handling."""
    error_message = "I failed"

    class SM(SobieskiMission):
        def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
            raise ValueError(error_message)

    (
        _path_to_discipline,
        _,
        _,
        discipline,
    ) = discipline_and_data
    discipline = SM()

    outputs_path = Path("outputs.pckl")
    return_code = _run_discipline_save_outputs(
        discipline,
        discipline.io.input_grammar.defaults,
        outputs_path,
        False,
        False,
        (),
        (),
    )
    assert return_code == 1
    with outputs_path.open("rb") as discipline_file:
        error, _ = pickle.load(discipline_file)
        assert isinstance(error, ValueError)
        assert error.args[0] == error_message


def test_main() -> None:
    """Test the main entry point."""
    with pytest.raises(SystemExit):
        main()


def test_cli_options_error(tmp_wd):
    """Verify that linearize options are consistent."""
    Path("dummy").touch()
    # --execute-at-linearize must be used with --linearize
    match = "The option --execute-at-linearize cannot be used without --linearize"
    with pytest.raises(ValueError, match=match):
        _parse_inputs((
            "dummy",
            "dummy",
            "dummy",
            "--execute-at-linearize",
        ))


@pytest.mark.parametrize("protocol", [pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL])
def test_path_serialization(tmp_wd, protocol):
    """Test the execution of a serialized discipline that contains Paths."""
    path_to_discipline = "discipline.pckl"
    discipline = PathDiscipline(tmp_wd)
    to_pickle(discipline, path_to_discipline)
    path_to_outputs = "outputs.pckl"
    path_to_input_data = "inputs.pckl"

    with open(path_to_input_data, "wb") as outf:
        pickler = pickle.Pickler(outf, protocol=protocol)
        pickler.dump((discipline.io.input_grammar.defaults, (), ()))

    completed = subprocess.run(
        f"gemseo-deserialize-run {path_to_discipline} "
        f"{path_to_input_data} {path_to_outputs}",
        shell=True,
        capture_output=True,
        cwd=tmp_wd,
    )

    assert completed.returncode == 0

    out = discipline.execute()
    assert out["y"] == 1
    assert out["y"] == 1
