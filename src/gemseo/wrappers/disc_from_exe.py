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
from ast import literal_eval
from collections.abc import Mapping
from collections.abc import MutableSequence
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Callable
from typing import Union

from numpy import array
from numpy import ndarray
from strenum import StrEnum

from gemseo.core.data_processor import DataProcessor
from gemseo.core.data_processor import FloatDataProcessor
from gemseo.utils.directory_creator import DirectoryNamingMethod
from gemseo.wrappers._base_disc_from_exe import _BaseDiscFromExe
from gemseo.wrappers._base_executable_runner import _BaseExecutableRunner

if TYPE_CHECKING:
    from gemseo.core.discipline_data import Data

LOGGER = logging.getLogger(__name__)

NUMERICS = [str(j) for j in range(10)]
INPUT_REGEX = r"GEMSEO_INPUT\{(.*)\}"
OUTPUT_REGEX = r"GEMSEO_OUTPUT\{(.*)\}"


class Parser(StrEnum):
    """Built-in parser types."""

    KEY_VALUE = "KEY_VALUE"
    """The output file is expected to have a key-value structure for each line."""

    TEMPLATE = "TEMPLATE"
    """The output is expected as a JSON file with the following format:

     .. code::

    {
       "a": GEMSEO_OUTPUT{a::1.0},
       "b": GEMSEO_OUTPUT{b::2.0},
       "c": GEMSEO_OUTPUT{c::3.0}
    }
    """


OutputParser = Callable[
    [Mapping[str, tuple[int]], Sequence[str]], Mapping[str, Union[ndarray, float]]
]
"""Output parser type."""

InputWriter = Callable[
    [
        Union[str, Path],
        Mapping[str, ndarray],
        Mapping[str, tuple[int, int, int]],
        MutableSequence[str],
    ],
    None,
]
"""Input writer type."""


class DiscFromExe(_BaseDiscFromExe):
    """Specific wrapper for executables.

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
    - For security reasons,
      the executable is executed via the Python subprocess library with no shell.
      If a shell is needed, you may override this in a derived class.
    """

    input_template: Path
    """The path to the input template file."""

    output_template: Path
    """The path to the output template file."""

    input_filename: str
    """The name of the input file."""

    output_filename: str
    """The name of the output file."""

    parse_outfile: OutputParser
    """The function used to parse the output template file."""

    write_input_file: InputWriter
    """The function used to write the input template file."""

    data_processor: DataProcessor
    """A data processor to be used before the execution of the discipline."""

    # TODO: API: remove use_shell.
    # TODO: API: rename executable_command into command_line
    # TODO: API: rename folders_iter into directory_naming_method
    # TODO: API: rename output_folder_basepath into root_directory
    def __init__(
        self,
        input_template: str | Path,
        output_template: str | Path,
        output_folder_basepath: str | Path,
        executable_command: str,
        input_filename: str | Path,
        output_filename: str | Path,
        folders_iter: DirectoryNamingMethod = DirectoryNamingMethod.NUMBERED,
        name: str | None = None,
        parse_outfile_method: Parser | OutputParser = Parser.TEMPLATE,
        write_input_file_method: InputWriter | None = None,
        parse_out_separator: str = "=",
        use_shell: bool = True,
        clean_after_execution: bool = False,
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
            folders_iter: The method to create the execution directories.
            parse_outfile_method: The optional method that can be provided
                by the user to parse the output template file.
                If the :attr:`~.Parser.KEY_VALUE` is used as
                output parser, the user may specify the separator key.
            write_input_file_method: The method to write the input data file.
                If ``None``,
                use :func:`~.write_input_file`.
            parse_out_separator: The separator used for the
                :attr:`~.Parser.KEY_VALUE` output parser.
            use_shell: This argument is ignored and will be removed,
                the shell is not used.
            output_folder_basepath: The base path of the execution directories.

        Raises:
            TypeError: If the provided ``parse_outfile_method`` is not callable.
                If the provided ``write_input_file_method`` is not callable.
        """  # noqa:D205 D212 D415
        if use_shell:
            LOGGER.warning(
                "The argument 'use_shell' is no longer used,"
                "the executable is run without shell."
            )
        executable_runner = _BaseExecutableRunner(
            root_directory=output_folder_basepath,
            command_line=executable_command,
            directory_naming_method=folders_iter,
        )
        super().__init__(
            executable_runner, name=name, clean_after_execution=clean_after_execution
        )

        self.input_template = Path(input_template)
        self.output_template = Path(output_template)
        self.input_filename = input_filename
        self.output_filename = output_filename

        if parse_outfile_method == Parser.TEMPLATE:
            self.parse_outfile = parse_outfile
        elif parse_outfile_method == Parser.KEY_VALUE:
            self.parse_outfile = lambda a, b: parse_key_value_file(
                a, b, parse_out_separator
            )
        else:
            self.parse_outfile = parse_outfile_method

        if not callable(self.parse_outfile):
            raise TypeError("The parse_outfile_method must be callable.")

        self.write_input_file = write_input_file_method or write_input_file
        if not callable(self.write_input_file):
            raise TypeError("The write_input_file_method must be callable.")

        self._out_pos = None
        self._input_data = None
        self._in_lines = None
        self._out_lines = None
        self.data_processor = FloatDataProcessor()
        self.__parse_templates_and_set_grammars()

    # TODO: API: rename as command_line
    @property
    def executable_command(self):
        """The executable command."""
        return self._executable_runner.command_line

    def __parse_templates_and_set_grammars(self) -> None:
        """Parse the templates and set the grammar of the discipline."""
        with self.input_template.open() as infile:
            self._in_lines = infile.readlines()
        with self.output_template.open() as outfile:
            self._out_lines = outfile.readlines()

        self._input_data, self._in_pos = parse_template(self._in_lines, True)
        input_names = self._input_data.keys()
        self.input_grammar.update_from_names(input_names)

        names_to_values, self._out_pos = parse_template(self._out_lines, False)
        output_names = names_to_values.keys()
        self.output_grammar.update_from_names(output_names)
        LOGGER.debug(
            "Initialize discipline from template. Input grammar: %s", input_names
        )
        LOGGER.debug(
            "Initialize discipline from template. Output grammar: %s", output_names
        )
        self.default_inputs = {
            k: array([literal_eval(v)]) for k, v in self._input_data.items()
        }

    def _create_inputs(self) -> None:
        """Write the input file."""
        self.write_input_file(
            self._executable_runner.last_execution_directory / self.input_filename,
            self.local_data,
            self._in_pos,
            self._in_lines,
        )

    def _parse_outputs(self) -> Data:
        """Parse the output file."""
        with (
            self._executable_runner.last_execution_directory / self.output_filename
        ).open() as outfile:
            out_lines = outfile.readlines()

        if len(out_lines) != len(self._out_lines):
            raise ValueError(
                "The number of lines of the output file changed."
                "This is not supported yet"
            )

        return self.parse_outfile(self._out_pos, out_lines)


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

    with Path(input_file_path).open("w") as infile_o:
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

    Raises:
        ValueError: If the amount of separators in the lines are not consistent with the
            keys and values.
            If the float values cannot be parsed.
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
            except BaseException:
                raise ValueError(f"Failed to parse value as float {value}.") from None

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
            if char in {"E", "e"}:
                if found_e:
                    break
                found_e = True
                continue
            # Check that we have not reached EOL or space or whatever
            if char not in NUMERICS:
                break
            if i == maxi - 1:
                break

        output_value = out_text[start:i]
        LOGGER.info("Parsed %s got output %s", output_name, output_value)
        values[output_name] = array([float(output_value)])

    return values
