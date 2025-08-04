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
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gemseo import from_pickle
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.deserialize_and_run import main
from gemseo.utils.pickle import to_pickle

if TYPE_CHECKING:
    from pytest import CaptureFixture  # noqa: PT013
    from pytest import MonkeyPatch  # noqa: PT013

    from gemseo.core.discipline.discipline_data import DisciplineData


def set_cli_args(monkeypatch: MonkeyPatch, args: str) -> None:
    """Monkey patch the sys.argv to simulate a command line call."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            # The program name.
            "dummy-name",
            *args.split(),
        ],
    )


def check_main_error(
    capsys: CaptureFixture[str],
    error_msg: str,
) -> None:
    """Check that the main function exits with an error code and
    gives the expected message."""
    with pytest.raises(SystemExit, match="2"):
        main()
    # The replacement is for the case that used protocol_999 where 999 may be wrapped
    # with ' depending on the python version and the platform.
    assert (
        capsys.readouterr()
        .err.strip()
        .replace("'", "")
        .endswith(error_msg.replace("'", ""))
    )


@pytest.mark.parametrize(
    ("arg_name", "args"),
    [
        (
            "discipline_path",
            "dummy1",
        ),
        (
            "inputs_path",
            "dummy1 dummy2",
        ),
    ],
)
def test_cli_input_file_error(
    tmp_wd: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    arg_name: str,
    args: str,
):
    """Verify the cli error related to file existence."""
    # Create dummy files but for the first one, which will be missing.
    for file_ in args.split()[:-1]:
        Path(file_).touch()

    bad_file_name = args.split()[-1]
    error_msg = (
        f"dummy-name: error: argument {arg_name}: can't open '{bad_file_name}': "
        f"[Errno 2] No such file or directory: '{bad_file_name}'"
    )

    set_cli_args(monkeypatch, args)
    check_main_error(capsys, error_msg)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (
            "--execute-at-linearize",
            "The option --execute-at-linearize cannot be used without --linearize",
        ),
        (
            "--protocol dummy",
            "error: argument --protocol: invalid int value: 'dummy'",
        ),
        (
            "--protocol 999",
            "error: argument --protocol: invalid choice: 999"
            " (choose from 0, 1, 2, 3, 4, 5)",
        ),
    ],
)
def test_cli_protocol_error(
    tmp_wd: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    args: str,
    error: str,
):
    """Verify the cli errors."""
    # Create a dummy file to pass the checks on the input files.
    Path("dummy").touch()
    set_cli_args(monkeypatch, "dummy dummy dummy " + args)
    check_main_error(capsys, error)


@pytest.mark.parametrize(
    "linearize",
    [
        "",
        "--linearize",
        "--linearize --execute-at-linearize",
    ],
)
@pytest.mark.parametrize(
    "protocol",
    [
        "",
        f"--protocol {pickle.DEFAULT_PROTOCOL}",
        f"--protocol {pickle.HIGHEST_PROTOCOL}",
    ],
)
def test_main(
    monkeypatch: MonkeyPatch,
    tmp_wd: Path,
    linearize: str,
    protocol: str,
) -> None:
    """Test main."""
    disc_path = tmp_wd / "discipline.pickle"
    inputs_path = tmp_wd / "inputs.pickle"
    outputs_path = tmp_wd / "outputs.pickle"

    set_cli_args(
        monkeypatch,
        f"{disc_path!s} {inputs_path!s} {outputs_path!s} {linearize} {protocol}",
    )

    # Create the test data.
    disc = SobieskiMission()
    input_data = disc.io.input_grammar.defaults
    differentiated_inputs = ()
    differentiated_outputs = ()
    kwargs = {"protocol": int(protocol.split()[-1])} if protocol else {}
    to_pickle(disc, disc_path, **kwargs)
    to_pickle(
        (input_data, differentiated_inputs, differentiated_outputs),
        inputs_path,
        **kwargs,
    )

    # Execute the main entry point.
    assert not main()

    # Create the reference data.
    disc = SobieskiMission()

    if not linearize:
        ref_data = disc.execute(input_data)
        ref_jac = {}
    else:
        disc.add_differentiated_inputs(differentiated_inputs)
        disc.add_differentiated_outputs(differentiated_outputs)
        ref_jac = disc.linearize(
            input_data,
            execute="--execute-at-linearize" in linearize,
        )
        ref_data = disc.io.data

    # Compare the outputs.
    data, jac = from_pickle(outputs_path)
    assert compare_dict_of_arrays(ref_data, data)
    assert compare_dict_of_arrays(ref_jac, jac)


class CrashingDiscipline(SobieskiMission):
    error_message = "This discipline is crashing"

    def _run(self, input_data: DisciplineData) -> None:
        raise ValueError(self.error_message)


def test_discipline_exception(tmp_wd: Path, monkeypatch: MonkeyPatch) -> None:
    """Test the outputs saving error handling."""
    disc = CrashingDiscipline()

    disc_path = tmp_wd / "discipline.pickle"
    inputs_path = tmp_wd / "inputs.pickle"
    outputs_path = tmp_wd / "outputs.pickle"

    to_pickle(disc, disc_path)
    to_pickle(
        (disc.io.input_grammar.defaults, (), ()),
        inputs_path,
    )

    set_cli_args(
        monkeypatch,
        f"{disc_path!s} {inputs_path!s} {outputs_path!s}",
    )

    assert main() == 1

    error, tb = from_pickle(outputs_path)

    assert isinstance(error, ValueError)
    assert str(error) == disc.error_message

    assert tb.startswith("Traceback")
    assert tb.endswith("ValueError: This discipline is crashing")
