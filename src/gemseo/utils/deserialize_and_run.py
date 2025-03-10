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
"""Executable that deserializes a discipline and executes it."""

from __future__ import annotations

import argparse
import pickle
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.typing import JacobianData


def _parse_inputs(
    args: Sequence[str] = (),
) -> tuple[Path, Path, Path, bool, bool, int]:
    """Parse the arguments of the command.

    Args:
        args: The command line arguments. If empty, uses ``sys.argv[1:]``.

    Returns:
        The path to the serialized discipline, the path to the serialized input data,
        the path to the serialized output data, whether to linearize the discipline,
        wether to call :meth:`.Discipline.execute` when calling
        :meth:`.Discipline.linearize`,
        and the pickle protocol to use for serializing outputs.

    Raises:
        RuntimeError: When one of the paths provided in the arguments does not exist,
            or an invalid number of arguments are passed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Deserialize the inputs, run the discipline "
            "and serialize the output to disk."
        ),
    )
    parser.add_argument(
        "discipline_path",
        help="The path to the serialized discipline.",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "inputs_path",
        help="The path to the serialized input data.",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "outputs_path", help="The path to the serialized output data.", type=Path
    )
    parser.add_argument(
        "--linearize",
        help="Whether to linearize the discipline or execute.",
        action="store_true",
    )
    parser.add_argument(
        "--execute-at-linearize",
        help="""Whether to call execute() when calling linearize().
        Requires --linearize.""",
        action="store_true",
    )
    parser.add_argument(
        "--protocol",
        help="""The pickle protocol to use for serializing outputs.""",
        type=int,
        default=pickle.HIGHEST_PROTOCOL,
    )

    parsed_args = parser.parse_args(args or None)

    if parsed_args.execute_at_linearize and not parsed_args.linearize:
        msg = "The option --execute-at-linearize cannot be used without --linearize"
        raise ValueError(msg)

    return (
        Path(parsed_args.discipline_path.name),
        Path(parsed_args.inputs_path.name),
        Path(parsed_args.outputs_path),
        parsed_args.linearize,
        parsed_args.execute_at_linearize,
        parsed_args.protocol,
    )


def _run_discipline_save_outputs(
    discipline: Discipline,
    input_data: DisciplineData,
    outputs_path: Path,
    linearize: bool,
    execute_at_linearize: bool,
    differentiated_inputs: Iterable[str],
    differentiated_outputs: Iterable[str],
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> int:
    """Run or linearize the discipline and serialize its outputs to disk.

    Args:
        discipline: The discipline to run.
        input_data: The input data for the discipline.
        outputs_path: The path to the output data.
        linearize: Whether to linearize the discipline.
        execute_at_linearize: Whether to call execute() when calling linearize().
        differentiated_inputs: If the linearization is performed, the
            inputs that define the rows of the jacobian.
        differentiated_outputs: If the linearization is performed, the
            outputs that define the columns of the jacobian.
        protocol: The protocol to use for pickling.

    Returns:
        The return code, 0 if success, 1 if failure.
    """
    outputs: tuple[DisciplineData, JacobianData] | tuple[BaseException, str]

    try:
        if linearize:
            discipline.add_differentiated_inputs(differentiated_inputs)
            discipline.add_differentiated_outputs(differentiated_outputs)
            discipline.linearize(input_data, execute=execute_at_linearize)
            outputs = discipline.io.data, discipline.jac
        else:
            outputs = (discipline.execute(input_data), {})
            if discipline._has_jacobian:
                outputs = (outputs[0], discipline.jac)
    except BaseException as error:  # noqa: BLE001
        trace = traceback.format_exc()
        outputs = (error, trace)
        return_code = 1
    else:
        return_code = 0

    with outputs_path.open("wb") as file_:
        pickler = pickle.Pickler(file_, protocol=protocol)
        pickler.dump(outputs)

    return return_code


def main() -> int:
    """Deserialize the inputs, run the discipline and saves the output to the disk.

    Takes the input parameters from sys.argv:
        - discipline_path: The path to the serialized discipline.
        - inputs_path: The path to the serialized input data.
        - outputs_path: The path to the serialized output data.
        - --linearize: Whether to linearize the discipline or execute.
        - --execute-at-linearize: Whether to call execute() when calling linearize().
        - --protocol: The pickle protocol to use for serializing outputs.

    Returns:
        The return code, 0 if success, 1 if failure.

    Raises:
        RuntimeError: When one of the paths provided in the arguments does not exist,
            or an invalid number of arguments are passed.
    """
    (
        serialized_disc_path,
        input_data_path,
        outputs_path,
        linearize,
        execute_at_linearize,
        protocol,
    ) = _parse_inputs()

    with serialized_disc_path.open("rb") as discipline_file:
        discipline = pickle.load(discipline_file)

    with input_data_path.open("rb") as input_data_file:
        input_data, linearize_inputs, linearize_outputs = pickle.load(input_data_file)

    return _run_discipline_save_outputs(
        discipline,
        input_data,
        outputs_path,
        linearize,
        execute_at_linearize,
        linearize_inputs,
        linearize_outputs,
        protocol,
    )
