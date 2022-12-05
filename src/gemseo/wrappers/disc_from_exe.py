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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Make a discipline from an executable."""
from __future__ import annotations

import logging
import re
import subprocess
import sys
from ast import literal_eval
from copy import deepcopy
from multiprocessing import Lock
from multiprocessing import Value
from os import listdir
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import MutableSequence
from typing import Sequence
from uuid import uuid1

from numpy import array
from numpy import ndarray

from gemseo.core.data_processor import DataProcessor  # noqa: F401
from gemseo.core.data_processor import FloatDataProcessor
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.base_enum import BaseEnum

LOGGER = logging.getLogger(__name__)

NUMERICS = [str(j) for j in range(10)]
INPUT_REGEX = r"GEMSEO_INPUT\{(.*)\}"
OUTPUT_REGEX = r"GEMSEO_OUTPUT\{(.*)\}"


class FoldersIter(BaseEnum):
    NUMBERED = 0
    UUID = 1


class Parsers(BaseEnum):
    KEY_VALUE_PARSER = 0
    TEMPLATE_PARSER = 1
    CUSTOM_CALLABLE = 2


class DiscFromExe(MDODiscipline):
    """Generic wrapper for executables.

    This :class:`.MDODiscipline` uses template files
    describing the input and output variables.

    The templates can be generated with a graphical user interface (GUI).
    by executing the module :mod:`~gemseo.wrappers.template_grammar_editor`.

    An input template file is a JSON file formatted as

    .. code::

       {
          "a": GEMSEO_INPUT{a::1.0},
          "b": GEMSEO_INPUT{b::2.0},
          "c": GEMSEO_INPUT{c::3.0}
       }

    where ``"a"`` is the name of an input, and ``1.0`` is its default value.
    Similarly,
    an output template file is a JSON file formatted as

    .. code::

       {
          "a": GEMSEO_OUTPUT{a::1.0},
          "b": GEMSEO_OUTPUT{b::2.0},
          "c": GEMSEO_OUTPUT{c::3.0}
       }

    where ``"a"`` is the name of an output, and ``1.0`` is its default value.

    The current limitations are

    - Only one input and one output template file, otherwise,
      inherit from this class and modify the parsers.
      Only limited input writing and output parser strategies
      are implemented. To change that, you can pass custom parsing and
      writing methods to the constructor.
    - The only limitation in the current file format is that
      it must be a plain text file and not a binary file.
      In this case, the way of interfacing it is
      to provide a specific parser to the :class:`.DiscFromExe`,
      with the :func:`.write_input_file_method`
      and :func:`.parse_outfile_method` arguments of the constructor.
    """

    input_template: str
    """The path to the input template file."""

    output_template: str
    """The path to the output template file."""

    input_filename: str
    """The name of the input file."""

    output_filename: str
    """The name of the output file."""

    executable_command: str
    """The executable command."""

    parse_outfile: Callable[[Mapping[str, tuple[int]]], Sequence[str]]
    """The function used to parse the output template file."""

    write_input_file: Callable[
        [str, Mapping[str, ndarray], Mapping[str, tuple[int]], Sequence[int]], str
    ]
    """The function used to write the input template file."""

    output_folder_basepath: str
    """The base path of the execution directories."""

    data_processor: DataProcessor
    """A data processor to be used before the execution of the discipline."""

    def __init__(
        self,
        input_template: str,
        output_template: str,
        output_folder_basepath: str,
        executable_command: str,
        input_filename: str,
        output_filename: str,
        folders_iter: str | FoldersIter = FoldersIter.NUMBERED,
        name: str | None = None,
        parse_outfile_method: str | Parsers = Parsers.TEMPLATE_PARSER,
        write_input_file_method: str | None = None,
        parse_out_separator: str = "=",
        use_shell: bool = True,
    ) -> None:
        """
        Args:
            input_template: The path to the input template file.
                The input locations in the file are marked
                by ``GEMSEO_INPUT{input_name::1.0}``,
                where ``input_name`` is the name of the input variable,
                and ``1.0`` is its default value.
            output_template: The path to the output template file.
                The output locations in the file are marked
                by ``GEMSEO_OUTPUT{output_name::1.0}``,
                where ``output_name`` is the name of the output variable,
                and ``1.0`` is its default value.
            executable_command: The command to run the executable.
                E.g. ``python my_script.py -i input.txt -o output.txt``
            input_filename: The name of the input file
                to be generated in the output folder.
                E.g. ``"input.txt"``.
            output_filename: The name of the output file
                to be generated in the output folder.
                E.g. ``"output.txt"``.
            folders_iter: The type of unique identifiers for the output folders.
                If :attr:`~.FoldersIter.NUMBERED`,
                the generated output folders will be ``f"output_folder_basepath{i+1}"``,
                where ``i`` is the maximum value
                of the already existing ``f"output_folder_basepath{i}"`` folders.
                Otherwise, a unique number based on the UUID function is
                generated. This last option shall be used if multiple MDO
                processes are run in the same work directory.
            parse_outfile_method: The optional method that can be provided
                by the user to parse the output template file.
                If ``None``,
                use :func:`~gemseo.wrappers.template_grammar_editor.parse_outfile`.
                If the :attr:`~.Parsers.KEY_VALUE_PARSER` is used as
                output parser, specify the separator key.
            write_input_file_method: The method to write the input data file.
                If ``None``,
                use :func:`~gemseo.wrappers.template_grammar_editor.write_input_file`.
            parse_out_separator: The separator used for the output parser.
            use_shell: If ``True``, run the command using the default shell.
                Otherwise, run directly the command.

        Raises:
            TypeError: If the provided ``write_input_file_method`` is not callable.
        """
        super().__init__(name=name)
        self.input_template = input_template
        self.output_template = output_template
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.executable_command = executable_command
        self.__use_shell = use_shell
        if (
            parse_outfile_method is None
            or parse_outfile_method == Parsers.TEMPLATE_PARSER
        ):
            self.parse_outfile = parse_outfile
            self._parse_outfile_method = Parsers.TEMPLATE_PARSER
        elif parse_outfile_method == Parsers.KEY_VALUE_PARSER:
            self.parse_outfile = lambda a, b: parse_key_value_file(
                a, b, parse_out_separator
            )
            self._parse_outfile_method = Parsers.KEY_VALUE_PARSER
        else:
            self.parse_outfile = parse_outfile_method
            self._parse_outfile_method = Parsers.CUSTOM_CALLABLE

        if not callable(self.parse_outfile):
            raise TypeError("The parse_outfile_method must be callable")

        self.write_input_file = write_input_file_method or write_input_file
        if not callable(self.write_input_file):
            raise TypeError("The write_input_file_method must be callable.")

        self.__lock = Lock()
        self.__folders_iter = None
        self.folders_iter = folders_iter
        self.output_folder_basepath = output_folder_basepath
        self.__check_base_path_on_windows()
        self._out_pos = None
        self._input_data = None
        self._output_data = None
        self._in_lines = None
        self._out_lines = None
        self._counter = Value("i", self._get_max_outdir())
        self.data_processor = FloatDataProcessor()
        self.__parse_templates_and_set_grammars()

    @property
    def folders_iter(self) -> FoldersIter:
        """The names of the new execution directories.

        Raises:
            ValueError: When the value is not a valid :class:`.FoldersIter`.
        """
        return self.__folders_iter

    @folders_iter.setter
    def folders_iter(
        self,
        value: str | FoldersIter,
    ) -> None:
        if value not in FoldersIter:
            raise ValueError(f"{value} is not a valid FoldersIter value.")
        self.__folders_iter = FoldersIter[value]

    def __check_base_path_on_windows(self) -> None:
        """Check that the base path can be used.

        Raises:
            ValueError: When the users use the shell under Windows
            and the base path is located on a network location.
        """
        if sys.platform.startswith("win") and self.__use_shell:
            output_folder_base_path = Path(self.output_folder_basepath).resolve()
            if not output_folder_base_path.parts[0].startswith("\\\\"):
                return

            raise ValueError(
                "A network base path and use_shell cannot be used together"
                " under Windows, as cmd.exe cannot change the current directory"
                " to a UNC path."
                " Please try use_shell=False or use a local base path."
            )

    def __parse_templates_and_set_grammars(self) -> None:
        """Parse the templates and set the grammar of the discipline."""
        with open(self.input_template) as infile:
            self._in_lines = infile.readlines()
        with open(self.output_template) as outfile:
            self._out_lines = outfile.readlines()

        self._input_data, self._in_pos = parse_template(self._in_lines, True)
        input_names = self._input_data.keys()
        self.input_grammar.update(input_names)

        names_to_values, self._out_pos = parse_template(self._out_lines, False)
        output_names = names_to_values.keys()
        self.output_grammar.update(output_names)
        LOGGER.debug(
            "Initialize discipline from template. Input grammar: %s", input_names
        )
        LOGGER.debug(
            "Initialize discipline from template. Output grammar: %s", output_names
        )
        self.default_inputs = {
            k: array([literal_eval(v)]) for k, v in self._input_data.items()
        }

    def _run(self) -> None:
        out_dir = Path(self.output_folder_basepath) / self.generate_uid()
        out_dir.mkdir()
        self.write_input_file(
            out_dir / self.input_filename,
            self.local_data,
            self._in_pos,
            self._in_lines,
        )

        if self.__use_shell:
            executable_command = self.executable_command
        else:
            executable_command = self.executable_command.split()

        err = subprocess.call(
            executable_command,
            shell=self.__use_shell,
            stderr=subprocess.STDOUT,
            cwd=out_dir,
        )
        if err != 0:
            raise RuntimeError(f"Execution failed and returned error code: {err}.")

        with open(out_dir / self.output_filename) as outfile:
            out_lines = outfile.readlines()

        if len(out_lines) != len(self._out_lines):
            raise ValueError(
                "The number of lines of the output file changed."
                "This is not supported yet"
            )

        self.local_data.update(self.parse_outfile(self._out_pos, out_lines))

    def generate_uid(self) -> str:
        """Generate a unique identifier for the execution directory.

        Generate a unique identifier for the current execution.
        If the folders_iter strategy is :attr:`~.FoldersIter.NUMBERED`,
        the successive iterations are named by an integer 1, 2, 3 etc.
        This is multiprocess safe.
        Otherwise, a unique number based on the UUID function is generated.
        This last option shall be used if multiple MDO processes are runned
        in the same workdir.

        Returns:
            An unique string identifier (either a number or a UUID).
        """
        if self.folders_iter == FoldersIter.NUMBERED:
            with self.__lock:
                self._counter.value += 1
                return str(self._counter.value)
        elif self.folders_iter == FoldersIter.UUID:
            return str(uuid1()).split("-")[-1]
        else:
            raise ValueError(
                f"{self.folders_iter} is not a valid method "
                "for creating the execution directories."
            )

    def _list_out_dirs(self) -> list[str]:
        """Return the directories in the output folder path.

        Returns:
             The list of the directories in the output folder path.
        """
        return listdir(self.output_folder_basepath)

    def _get_max_outdir(self) -> int:
        """Get the maximum current index of output folders.

        Returns:
             The maximum index in the output folders.
        """
        outs = list(self._list_out_dirs())
        if not outs:
            return 0
        return max(literal_eval(n) for n in outs)


def parse_template(
    template_lines: Sequence[str],
    grammar_is_input: bool,
) -> tuple[dict[str, str], dict[str, tuple[int, int, int]]]:
    """Parse the input or output template.

    This function parses the input (or output) template.
    It returns the tuple (names_to_values, names_to_positions), where:

    - `names_to_values` is the `{name:value}` dict:

       - name is the data name
       - value is the parsed input or output value in the template

    - `names_to_positions` describes the template format {name:(start,end,line_number)}:

       - `name` is the name of the input data
       - `start` is the index of the starting point in the input file template.
         This index is a line index (character number on the line)
       - `end` is the index of the end character in the template
       - `line_number` is the index of the line in the file

    Args:
        template_lines: The lines of the template file.
        grammar_is_input: Whether the template file describes input variables.

    Returns:
        A data structure containing the parsed input or output template.
    """
    regex = re.compile(INPUT_REGEX if grammar_is_input else OUTPUT_REGEX)
    # , re.MULTILINE
    names_to_values = {}
    names_to_positions = {}
    for line_index, line in enumerate(template_lines):
        for match in regex.finditer(line):
            name, value = match.groups()[0].split("::")
            names_to_values[name] = value
            if grammar_is_input:
                # If input mode: erase the template value.
                start, end = match.start(), match.end()
            else:
                # If output mode: catch all the output length and not more.
                start = match.start()
                end = start + len(value)

            names_to_positions[name] = (start, end, line_index)

    return names_to_values, names_to_positions


def write_input_file(
    input_file_path: str | Path,
    data: Mapping[str, ndarray],
    input_positions: Mapping[str, tuple[int, int, int]],
    input_lines: MutableSequence[str],
    float_format: str = "{:1.18g}",
) -> None:
    """Write the input file template from the input data.

    Args:
        input_file_path: The absolute path to the file to be written.
        data: The local data of the discipline.
        input_positions: The positions of the input variables,
            formatted as ``{"name": (start, end, line_number)}``,
            where ``"name"`` is the name of the input variable,
            ``start`` is the index of the starting point in the file,
            ``end`` is the index of the end character in the file,
            and ``line_number`` is the index of the line in the file.
            An index is a line index, i.e. a character number on the line.
        input_lines: The lines of the file.
        float_format: The format of the input data in the file.
    """
    f_text = deepcopy(input_lines)
    for input_name, (start, end, line_number) in input_positions.items():
        cline = f_text[line_number]
        f_text[line_number] = (
            cline[:start] + float_format.format(data[input_name]) + cline[end:]
        )

    with open(input_file_path, "w") as infile_o:
        infile_o.writelines(f_text)


def parse_key_value_file(
    _,
    out_lines: Sequence[str],
    separator: str = "=",
) -> dict[str, float]:
    """Parse the output file from the expected text positions.

    Args:
        out_lines: The lines of the output file template.
        separator: The separating characters of the key=value format.

    Returns:
        The output data in `.MDODiscipline` friendly data structure.
    """
    data = {}
    for line in out_lines:
        if separator in line:
            key_and_value = line.strip().split(separator)
            if len(key_and_value) != 2:
                raise ValueError(f"unbalanced = in line {line}.")

            key, value = key_and_value
            try:
                data[key.strip()] = float(literal_eval(value.strip()))
            except Exception:
                raise ValueError(f"Failed to parse value as float {value}.")

    return data


def parse_outfile(
    output_positions: Mapping[str, tuple[int]],
    out_lines: Sequence[str],
) -> dict[str, ndarray]:
    """Parse the output template file from the expected text positions.

    Args:
        output_positions: The output position for each output variable,
            specified as ``{"name": (start, end, line_number)}``,
            where ``"name"`` is the name of the output variable,
            ``start`` is the index of the starting point in the file,
            ``end`` is the index of the end character in the file,
            and ``line_number`` is the index of the line in the file.
            An index is a line index, i.e. a character number on the line.
        out_lines: The lines of the file.

    Returns:
        The output data.
    """
    values = {}
    for output_name, (start, _, line_number) in output_positions.items():
        found_dot = False
        found_e = False
        # In case generated files has fewer lines
        if line_number > len(out_lines) - 1:
            break
        out_text = out_lines[line_number]
        i = start
        maxi = len(out_text)
        while True:
            # The problem is that the output file used for the template may be
            # using an output that is longer or shorter than the one generated
            # at runtime. We must find the proper end of the expression...
            i += 1
            char = out_text[i]
            if char == ".":
                # We found the . in float notation
                if found_dot or found_e:
                    break
                found_dot = True
                continue
            # We found the e in exp notation
            if char in ("E", "e"):
                if found_e:
                    break
                found_e = True
                continue
            # Check that we have nout reached EOL or space or whatever
            if char not in NUMERICS:
                break
            if i == maxi - 1:
                break

        output_value = out_text[start:i]
        LOGGER.info("Parsed %s got output %s", output_name, output_value)
        values[output_name] = array([float(output_value)])

    return values
