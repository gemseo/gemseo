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
"""Excel file parser for the study analyses."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Final

from pandas import DataFrame
from pandas import read_excel

from gemseo import get_available_formulations
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable

LOGGER = logging.getLogger(__name__)


class XLSStudyParser:
    """A study specification based on an Excel file.

    The Excel file shall contain one sheet per discipline:

    - the name of the sheet shall have the discipline name,
    - the sheet shall define the input names of the discipline
      as a vertical succession of cells starting with ``"Inputs"``:

        .. table:: Inputs

            +--------------+
            | Inputs       |
            +--------------+
            | input_name_1 |
            +--------------+
            | ...          |
            +--------------+
            | input_name_N |
            +--------------+

    - the sheet shall define the output names of the discipline
      as a vertical succession of cells starting with ``"Outputs"``:

    .. table:: Outputs

            +---------------+
            | Outputs       |
            +---------------+
            | output_name_1 |
            +---------------+
            | ...           |
            +---------------+
            | output_name_N |
            +---------------+

    - the empty lines of the series ``Inputs`` and ``Outputs`` are ignored,
    - the sheet may contain other data, but these will not be taken into account.

    If ``has_scenario`` is ``True``,
    the Excel file shall contain one sheet per scenario
    with a name starting by ``Scenario``.
    Distributed formulations shall contain one sheet for the main scenario
    and one sheet per sub-scenario.

    A scenario sheet shall have the following columns:

    .. table:: Scenario1

        +------------------+--------------------+-------------+-------------+-------------+---------------+----------------+
        | Design variables | Objective function | Constraints | Disciplines | Formulation |    Options    | Options values |
        +==================+====================+=============+=============+=============+===============+================+
        |       in1        |       out1         |    out2     |    Disc1    |    MDF      |   tolerance   |       0.1      |
        +------------------+--------------------+-------------+-------------+-------------+---------------+----------------+
        |                  |                    |             |    Disc2    |             | main_mda_name |   MDAJacobi    |
        +------------------+--------------------+-------------+-------------+-------------+---------------+----------------+

    These columns must satisfy some constraints:

    - all of them are mandatory, even if empty for the constraints,
    - their order does not matter,
    - one and only one formulation must be declared,
    - at least one objective must be provided,
    - at least one design variable must be provided,
    - all the objective functions and constraints must be outputs of a discipline,
      not necessarily the one of the current sheet,
    - all the design variables must be inputs of a discipline,
      not necessarily the one of the current sheet.

    The columns ``Options`` and ``Options values`` are used
    to pass the formulation options.
    Note that for string type ``Option values``,
    the value can be written with or without the ``""`` characters.

    To use multi-level MDO formulations,
    create multiple scenarios,
    and add the name of the sub-scenarios
    in the list of disciplines of the main (system) scenario.

    An arbitrary number of levels can be generated this way
    (three, four, ..., n, level formulations).
    """  # noqa: E501

    xls_study_path: str
    """The path to the Excel file."""

    worksheets: dict[str, DataFrame]
    """The worksheets of the Excel file."""

    disciplines: dict[str, MDODiscipline]
    """The non-executable disciplines."""

    scenarios: dict[str, dict[str, str | list[str]]]
    """The descriptions of the scenarios."""

    inputs: set[str]
    """The names of the input variables."""

    outputs: set[str]
    """The names of the output variables."""

    SCENARIO_PREFIX: Final[str] = "Scenario"
    DISCIPLINE: Final[str] = "Discipline"
    DISCIPLINES: Final[str] = "Disciplines"
    OBJECTIVE_FUNCTION: Final[str] = "Objective function"
    CONSTRAINTS: Final[str] = "Constraints"
    DESIGN_VARIABLES: Final[str] = "Design variables"
    FORMULATION: Final[str] = "Formulation"
    OPTIONS: Final[str] = "Options"
    OPTION_VALUES: Final[str] = "Options values"
    __INPUTS: Final[str] = "Inputs"
    __OUTPUTS: Final[str] = "Outputs"
    __SPACE: Final[str] = MultiLineString.INDENTATION

    def __init__(self, xls_study_path: str, has_scenario: bool = True) -> None:
        """Args:
            xls_study_path: The path to the Excel file describing the study.
            has_scenario: Whether the Excel file has a scenario sheet.

        Raises:
            IOError: If the Excel file cannot be opened.
            ValueError: If no scenario has been found in Excel file
                while the study is an MDO one.
        """  # noqa: D205 D212 D415
        self.xls_study_path = xls_study_path
        try:
            self.worksheets = read_excel(
                xls_study_path, sheet_name=None, engine="openpyxl"
            )
        except OSError:
            LOGGER.exception("Failed to open the study file: %s", xls_study_path)
            raise

        self.__log_number_objects_detected(True)
        self.disciplines = {}
        self.scenarios = {}
        self.inputs = set()
        self.outputs = set()

        self._init_disciplines()
        self.__set_scenario_descriptions()

        if has_scenario and not self.scenarios:
            raise ValueError("No scenario found in the XLS file.")

    def _init_disciplines(self) -> None:
        """Initialize the disciplines.

        Raises:
            ValueError: If the discipline has no input column or output column.
        """
        all_inputs = []
        all_outputs = []
        string = MultiLineString()
        string.indent()
        missing_column_msg = "The sheet of the discipline '{}' must have a column '{}'"
        for sheet_name, sheet_value in self.worksheets.items():
            if sheet_name.startswith(self.SCENARIO_PREFIX):
                continue

            # We use add("{}", sheet_name) rather than add(sheet_name)
            # to prevent problems with special characters in disc_name,
            # e.g. "Discipline{1}".
            string.add("{}", sheet_name)
            try:
                inputs = self.__get_series(sheet_value, self.__INPUTS)
                all_inputs += inputs
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(sheet_name, self.__INPUTS)
                ) from None

            try:
                outputs = self.__get_series(sheet_value, self.__OUTPUTS)
                all_outputs += outputs
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(sheet_name, self.__OUTPUTS)
                ) from None

            discipline = MDODiscipline(sheet_name)
            discipline.input_grammar.update_from_names(inputs)
            discipline.output_grammar.update_from_names(outputs)
            string.indent()
            string.add("{}: {}", self.__INPUTS, pretty_str(inputs))
            string.add("{}: {}", self.__OUTPUTS, pretty_str(outputs))
            string.dedent()
            self.disciplines[sheet_name] = discipline

        LOGGER.info("%s", string)
        self.inputs = set(all_inputs)
        self.outputs = set(all_outputs)

    @staticmethod
    def __get_series(
        frame: DataFrame, series_name: str, raise_error: bool = True
    ) -> list[str]:
        """Return the data of a named column.

        Removes empty data.

        Args:
            frame: The pandas frame of the sheet.
            series_name: The name of the series.
            raise_error: Whether to raise a ``ValueError``
                when the series does not exist;
                otherwise, return an empty list.

        Returns:
            The names of the columns, if the series exist.

        Raises:
            ValueError: If the sheet has no name and ``raise_error`` is ``True``.
        """
        series = frame.get(series_name)
        if series is None:
            if raise_error:
                raise ValueError(f"The sheet has no series named '{series_name}'.")
            return []
        # Remove empty data
        return [val for val in series.tolist() if val == val]

    def __set_scenario_descriptions(self) -> None:
        """Define the descriptions of the different scenarios.

        In terms of objective function, the constraints and the design variables.

        Raises:
            ValueError: If at least one of following elements is missing:
                * ``disciplines`` column,
                * ``design variables`` column,
                * ``objectives`` column,
                * ``constraints`` column,
                * ``formulations`` column,
                * if a scenario has more than one formulation,
                * if a scenario has different number of option values.
        """
        self.scenarios = {}
        worksheets = self.__log_number_objects_detected(False)
        missing_column_msg = "Scenario {} has no {} column."
        for frame_name, frame in worksheets.items():
            try:
                disciplines = self.__get_series(frame, self.DISCIPLINES)
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(frame_name, self.DISCIPLINES)
                ) from None

            try:
                design_variables = self.__get_series(frame, self.DESIGN_VARIABLES)
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(frame_name, self.DESIGN_VARIABLES)
                ) from None

            try:
                objectives = self.__get_series(frame, self.OBJECTIVE_FUNCTION)
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(frame_name, self.OBJECTIVE_FUNCTION)
                ) from None

            try:
                constraints = self.__get_series(frame, self.CONSTRAINTS)
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(frame_name, self.CONSTRAINTS)
                ) from None

            try:
                formulation = self.__get_series(frame, self.FORMULATION)
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(frame_name, self.FORMULATION)
                ) from None

            options = self.__get_series(frame, self.OPTIONS, False)
            option_values = self.__get_series(frame, self.OPTION_VALUES, False)

            if len(formulation) != 1:
                raise ValueError(
                    "Scenario {} must have one {} value.".format(
                        str(frame_name), self.FORMULATION
                    )
                ) from None

            if options is not None and len(options) != len(option_values):
                raise ValueError(
                    f"Options {options} and Options values {option_values} "
                    "must have the same length."
                ) from None

            scenario_description = {
                self.DISCIPLINES: disciplines,
                self.OBJECTIVE_FUNCTION: objectives,
                self.CONSTRAINTS: constraints,
                self.DESIGN_VARIABLES: design_variables,
                self.FORMULATION: formulation[0],
                self.OPTIONS: options,
                self.OPTION_VALUES: option_values,
            }

            self.scenarios[frame_name] = scenario_description

        for scenario_name, scenario_description in self.scenarios.items():
            self.__check_scenario_description(
                scenario_description[self.OBJECTIVE_FUNCTION],
                scenario_description[self.CONSTRAINTS],
                scenario_description[self.DISCIPLINES],
                scenario_description[self.DESIGN_VARIABLES],
                scenario_description[self.FORMULATION],
                scenario_name,
            )

    def __log_number_objects_detected(
        self, is_discipline: bool
    ) -> dict[str | int, DataFrame]:
        """Log the number of worksheets matching a given type.

        Args:
            is_discipline: Whether the worksheet defines a discipline;
                otherwise, a scenario.

        Returns:
            The worksheets defining a discipline if ``is_discipline`` is ``True``;
            otherwise the others that are supposed to define scenarios.
        """
        worksheets = {
            sheet_name: sheet_value
            for sheet_name, sheet_value in self.worksheets.items()
            if sheet_name.startswith(self.SCENARIO_PREFIX) is not is_discipline
        }
        if worksheets:
            n_worksheets = len(worksheets)
            LOGGER.info(
                "%s %s%s detected",
                n_worksheets,
                "discipline" if is_discipline else "scenario",
                "s" if n_worksheets > 1 else "",
            )

        return worksheets

    def __check_scenario_description(
        self,
        objectives: Iterable[str],
        constraints: Iterable[str],
        disciplines: Iterable[str],
        design_variables: Iterable[str],
        formulation: str,
        scenario_name: str,
    ) -> None:
        """Checks the optimization problem consistency.

        Args:
            objectives: The names of the objectives.
            constraints: The names of the constraints.
            disciplines: The names of the disciplines.
            design_variables: The names of the design variables.
            formulation: The name of the MDO formulation.
            scenario_name: The name of the scenario.

        Raises:
            ValueError: If at least one of following situation happens:
                * design variables in the scenario are not input of any discipline,
                * some disciplines do not exist in the scenario,
                * some constraints are not outputs of any discipline,
                * the objective function is not an output of any discipline,
                * the formulation is unknown.
        """
        string = MultiLineString()
        string.indent()
        # We use add("{}", scn_name) rather than add(scn_name)
        # to prevent problems with special characters in scn_name, e.g. "Scenario{1}".
        string.add("{}", scenario_name)
        string.indent()
        string.add("Objectives: {}", pretty_str(objectives))
        string.add("Disciplines: {}", pretty_str(disciplines))
        string.add("Constraints: {}", pretty_str(constraints))
        string.add("Design variables: {}", pretty_str(design_variables))
        string.add("Formulation: {}", formulation)
        LOGGER.info("%s", string)

        missing = set(design_variables) - self.inputs
        if missing:
            raise ValueError(
                f"{scenario_name}: some design variables are not "
                f"the inputs of any discipline: {missing}."
            )

        missing = set(disciplines) - set(self.disciplines.keys()) - set(self.scenarios)
        if missing:
            raise ValueError(
                f"{scenario_name}: some disciplines don't exist: {missing}."
            )

        missing = set(constraints) - self.outputs
        if missing:
            raise ValueError(
                f"{scenario_name}: some constraints are not "
                f"the outputs of any discipline: {missing}."
            )

        missing = set(objectives) - self.outputs
        if missing:
            raise ValueError(
                f"{scenario_name}: some objectives are not "
                f"the outputs of any discipline: {missing}."
            )
        if not objectives:
            raise ValueError(f"{scenario_name}: no objectives are defined")

        if formulation not in get_available_formulations():
            raise ValueError(
                f"{scenario_name}: unknown formulation '{formulation}'; "
                f"use one of: {get_available_formulations()}"
            )
