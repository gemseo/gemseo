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
from pickle import HIGHEST_PROTOCOL
from typing import TYPE_CHECKING

from gemseo.utils.pickle import from_pickle
from gemseo.utils.pickle import to_pickle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.typing import JacobianData


def _parse_inputs(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the arguments of the command line.

    Args:
        argv: The command line arguments, if any.

    Returns:
        The parsed arguments.
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
        type=argparse.FileType(),
    )
    parser.add_argument(
        "inputs_path",
        help="The path to the serialized input data.",
        type=argparse.FileType(),
    )
    parser.add_argument(
        "outputs_path",
        help="The path to the serialized output data.",
        type=Path,
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
        default=pickle.HIGHEST_PROTOCOL,
        type=int,
        choices=tuple(range(HIGHEST_PROTOCOL + 1)),
    )

    parsed_args = parser.parse_args(argv)

    if parsed_args.execute_at_linearize and not parsed_args.linearize:
        msg = "The option --execute-at-linearize cannot be used without --linearize"
        parser.error(msg)

    return parsed_args


def _execute_discipline(
    parsed_args: argparse.Namespace,
) -> tuple[DisciplineData, JacobianData]:
    """Execute or linearize the discipline and serialize its outputs to disk.

    Args:
        parsed_args: The parsed arguments from the command line.

    Returns:
        The discipline's data and jacobian.
    """
    # For paths that are checked for existence,
    # the parsed argument is not a path, we need to get it from the name attribute.
    discipline = from_pickle(parsed_args.discipline_path.name)
    input_data, linearize_inputs, linearize_outputs = from_pickle(
        parsed_args.inputs_path.name
    )

    if parsed_args.linearize:
        discipline.add_differentiated_inputs(linearize_inputs)
        discipline.add_differentiated_outputs(linearize_outputs)
        jac = discipline.linearize(input_data, execute=parsed_args.execute_at_linearize)
        data = discipline.io.data
    else:
        data = discipline.execute(input_data)
        jac = {} if not discipline._has_jacobian else discipline.jac

    return data, jac


def main(argv: Sequence[str] | None = None) -> int:
    """Deserialize the inputs, execute the discipline and save the output to the disk.

    Args:
        argv: The command line arguments.

    Returns:
        The return code, 0 on success, 1 on failure.
    """
    parsed_args = _parse_inputs(argv)

    outputs: tuple[DisciplineData, JacobianData] | tuple[BaseException, str]

    try:
        data, jac = _execute_discipline(parsed_args)
    except BaseException as error:  # noqa: BLE001
        outputs = (error, traceback.format_exc().strip())
        return_code = 1
    else:
        outputs = (data, jac)
        return_code = 0

    to_pickle(outputs, parsed_args.outputs_path, protocol=parsed_args.protocol)

    return return_code
