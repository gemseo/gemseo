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
import os
import pickle
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.core.discipline import MDODiscipline
    from gemseo.core.discipline_data import DisciplineData


def _parse_inputs(args: Iterable[str] | None = None) -> tuple[Path, Path, Path, Path]:
    """Parse the arguments of the command.

    Args:
        args: The command line arguments. If ``None``, uses sys.argv[1:]

    Returns:
        The path to the workdir, the path to the serialized discipline, the path
        to the serialized input data, the path to the serialized output data

    Raises:
        RuntimeError: When one of the paths provided in the arguments does not exist,
            or an invalid number of arguments are passed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Deserialize the inputs, run the discipline "
            "and saves the output to the disk."
        ),
    )
    parser.add_argument(
        "run_workdir",
        help="The path to the workdir where the files will be generated.",
        type=Path,
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

    parsed_args = parser.parse_args(args)

    workir_path = parsed_args.run_workdir
    if not workir_path.exists():
        raise FileNotFoundError(f"Work directory {workir_path} does not exist.")

    serialized_disc_path = Path(parsed_args.discipline_path.name)
    input_data_path = Path(parsed_args.inputs_path.name)

    return workir_path, serialized_disc_path, input_data_path, parsed_args.outputs_path


def _run_discipline_save_outputs(
    discipline: MDODiscipline,
    input_data: DisciplineData,
    outputs_path: Path,
    workdir_path: Path,
) -> int:
    """Run the discipline and save its outputs to the disk.

    Args:
        discipline: The discipline to run.
        input_data: The input data for the discipline.
        outputs_path: The path to the output data.
        workdir_path: The path to the working directory.

    Returns:
        The return code, 0 if success, 1 if failure.
    """
    cwd = Path.cwd()
    os.chdir(workdir_path)

    try:
        outputs = discipline.execute(input_data)
    except BaseException as error:
        trace = traceback.format_exc()
        outputs = (error, trace)
        return_code = 1
    else:
        return_code = 0

    with outputs_path.open("wb") as outfobj:
        pickler = pickle.Pickler(outfobj, protocol=2)
        pickler.dump(outputs)

    os.chdir(cwd)
    return return_code


def main() -> int:
    """Deserialize the inputs, run the discipline and saves the output to the disk.

    Takes the input parameters from sys.argv:
        - run_workdir: The path to the workdir where the files will be generated.
        - discipline_path: The path to the serialized discipline.
        - inputs_path: The path to the serialized input data.
        - outputs_path: The path to the serialized output data.

    Returns:
        The return code, 0 if success, 1 if failure.

    Raises:
        RuntimeError: When one of the paths provided in the arguments does not exist,
            or an invalid number of arguments are passed.
    """
    workir_path, serialized_disc_path, input_data_path, outputs_path = _parse_inputs()

    with serialized_disc_path.open("rb") as discipline_file:
        discipline = pickle.load(discipline_file)

    with input_data_path.open("rb") as input_data_file:
        input_data = pickle.load(input_data_file)

    return _run_discipline_save_outputs(
        discipline, input_data, outputs_path, workir_path
    )
