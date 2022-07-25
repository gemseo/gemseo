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
"""Generate a N2 and XDSM into files (and/or web page) from an Excel description of the
MDO problem."""
from __future__ import annotations

import logging
from ast import literal_eval
from typing import Iterable
from typing import Mapping

from pandas import DataFrame  # noqa F401
from pandas import read_excel

from gemseo.api import create_design_space
from gemseo.api import create_scenario
from gemseo.api import generate_n2_plot
from gemseo.api import get_available_formulations
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.mdo_scenario import MDOScenario

LOGGER = logging.getLogger(__name__)


class XLSStudyParser:
    """Parse the input Excel file that describe the GEMSEO study.

    The Excel file must contain one sheet per discipline.
    The name of the sheet shall have the name of the discipline.
    The sheet shall have at least two columns,
    one for the inputs and one for the outputs,
    with the following format:

    +--------+---------+----------------+--------------------+----------------+
    | Inputs | Outputs |Design variables| Objective function |  Constraints   |
    +========+=========+================+====================+================+
    |  in1   |  out1   |      in1       |       out1         |     out2       |
    +--------+---------+----------------+--------------------+----------------+
    |  in2   |  out2   |                |                    |                |
    +--------+---------+----------------+--------------------+----------------+

    Empty lines are ignored.
    All the objective functions and constraints must be outputs of a discipline,
    not necessarily the one of the current sheet.
    All the design variables must be inputs of a discipline,
    not necessarily the one of the current sheet.
    """

    xls_study_path: str
    """The path to the Excel file."""

    frames: dict[str, DataFrame]
    """The data frames created from the Excel file."""

    disciplines: dict[str, MDODiscipline]
    """The disciplines."""

    scenarios: dict[str, dict[str, str | list[str]]]
    """The descriptions of the scenarios parsed in the Excel file."""

    inputs: set[str]
    """The input variables."""

    outputs: set[str]
    """The output variables."""

    SCENARIO_PREFIX = "Scenario"
    DISCIPLINE = "Discipline"
    DISCIPLINES = "Disciplines"
    OBJECTIVE_FUNCTION = "Objective function"
    CONSTRAINTS = "Constraints"
    DESIGN_VARIABLES = "Design variables"
    FORMULATION = "Formulation"
    OPTIONS = "Options"
    OPTIONS_VALUES = "Options values"

    def __init__(self, xls_study_path: str) -> None:
        """Initialize the study from the Excel specification.

        Args:
            xls_study_path: The path to the Excel file describing the study.

        Raises:
            IOError: If the Excel file cannot be opened.
            ValueError: If no scenario has been found in Excel file.
        """
        self.xls_study_path = xls_study_path
        try:
            self.frames = read_excel(xls_study_path, sheet_name=None, engine="openpyxl")
        except OSError:
            LOGGER.error("Failed to open the study file: %s", xls_study_path)
            raise

        LOGGER.info("Detected the following disciplines: %s", list(self.frames.keys()))
        self.disciplines = dict()
        self.scenarios = dict()
        self.inputs = dict()
        self.outputs = dict()

        self._init_disciplines()
        self._get_opt_pb_descr()

        if not self.scenarios:
            raise ValueError("No scenario found in the xls file")

    def _init_disciplines(self) -> None:
        """Initialize the disciplines.

        Raises:
            ValueError: If the discipline has no input column or output column.
        """
        all_inputs = []
        all_outputs = []
        for disc_name, frame in self.frames.items():
            if disc_name.startswith(self.SCENARIO_PREFIX):
                continue
            LOGGER.info("Parsing discipline %s", disc_name)

            missing_column_msg = (
                "The sheet of the discipline '{}' must have a column '{}'"
            )

            try:
                inputs = self._get_frame_series_values(frame, "Inputs")
            except ValueError:
                raise ValueError(missing_column_msg.format(disc_name, "Inputs"))

            all_inputs += inputs
            try:
                outputs = self._get_frame_series_values(frame, "Outputs")
            except ValueError:
                raise ValueError(missing_column_msg.format(disc_name, "Outputs"))

            all_outputs += outputs
            disc = MDODiscipline(disc_name)
            disc.input_grammar.update(inputs)
            disc.output_grammar.update(outputs)
            LOGGER.info("Inputs: %s", inputs)
            LOGGER.info("Outputs: %s", outputs)

            self.disciplines[disc_name] = disc

        self.inputs = set(all_inputs)
        self.outputs = set(all_outputs)

    @staticmethod
    def _get_frame_series_values(
        frame: DataFrame,
        series_name: str,
        return_none: bool | None = False,
    ) -> None:
        """Return the data of a named column.

        Removes empty data.

        Args:
            frame: The pandas frame of the sheet.
            series_name: The name of the series.
            return_none: If the series does not exists, returns None
                instead of raising a ValueError.

        Returns:
            The list of a named column.

        Raises:
            ValueError: If the sheet has no name.
        """
        series = frame.get(series_name)
        if series is None:
            if return_none:
                return None
            raise ValueError(f"The sheet has no serie named '{series_name}'")
        # Remove empty data
        # pylint: disable=comparison-with-itself
        return [val for val in series.tolist() if val == val]

    def _get_opt_pb_descr(self) -> None:
        """Initialize the objective function, constraints and design_variables.

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
        self.scenarios = dict()

        for frame_name, frame in self.frames.items():
            if not frame_name.startswith(self.SCENARIO_PREFIX):
                continue
            LOGGER.info("Detected scenario in sheet: %s", frame_name)

            missing_column_msg = "Scenario {} has no {} column!"

            try:
                disciplines = self._get_frame_series_values(frame, self.DISCIPLINES)
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(frame_name, self.DISCIPLINES)
                )

            try:
                design_variables = self._get_frame_series_values(
                    frame, self.DESIGN_VARIABLES
                )
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(frame_name, self.DESIGN_VARIABLES)
                )

            try:
                objectives = self._get_frame_series_values(
                    frame, self.OBJECTIVE_FUNCTION
                )
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(frame_name, self.OBJECTIVE_FUNCTION)
                )

            try:
                constraints = self._get_frame_series_values(frame, self.CONSTRAINTS)
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(frame_name, self.CONSTRAINTS)
                )

            try:
                formulation = self._get_frame_series_values(frame, self.FORMULATION)
            except ValueError:
                raise ValueError(
                    missing_column_msg.format(frame_name, self.FORMULATION)
                )

            options = self._get_frame_series_values(frame, self.OPTIONS, True)
            options_values = self._get_frame_series_values(
                frame, self.OPTIONS_VALUES, True
            )

            if len(formulation) != 1:
                raise ValueError(
                    "Scenario {} must have one {} value!".format(
                        str(frame_name), self.FORMULATION
                    )
                )

            if options is not None:
                if len(options) != len(options_values):
                    raise ValueError(
                        "Options {} and Options values {} "
                        "must have the same length!".format(options, options_values)
                    )

            formulation = formulation[0]

            scn = {
                self.DISCIPLINES: disciplines,
                self.OBJECTIVE_FUNCTION: objectives,
                self.CONSTRAINTS: constraints,
                self.DESIGN_VARIABLES: design_variables,
                self.FORMULATION: formulation,
                self.OPTIONS: options,
                self.OPTIONS_VALUES: options_values,
            }

            self.scenarios[frame_name] = scn

        for name, desc in self.scenarios.items():
            self._check_opt_pb(
                desc[self.OBJECTIVE_FUNCTION],
                desc[self.CONSTRAINTS],
                desc[self.DISCIPLINES],
                desc[self.DESIGN_VARIABLES],
                desc[self.FORMULATION],
                name,
            )

    def _check_opt_pb(
        self,
        objectives: Iterable[str],
        constraints: Iterable[str],
        disciplines: Iterable[str],
        design_variables: Iterable[str],
        formulation: str,
        scn_name: str,
    ) -> None:
        """Checks the optimization problem consistency.

        Args:
            objectives: The names of the objectives.
            constraints: The names of the constraints.
            disciplines: The names of the disciplines.
            design_variables: The names of the design variables.
            formulation: The name of the MDO formulation.
            scn_name: The name of the scenario.

        Raises:
            ValueError: If at least one of following situation happens:
                * design variables in the scenario are not input of any discipline,
                * some disciplines do not exist in the scenario,
                * some constraints are not outputs of any discipline,
                * the objective function is not an output of any discipline,
                * the formulation is unknown.
        """
        LOGGER.info("New scenario: %s", scn_name)
        LOGGER.info("Objectives: %s", objectives)
        LOGGER.info("Disciplines: %s", disciplines)
        LOGGER.info("Constraints: %s", constraints)
        LOGGER.info("Design variables: %s", design_variables)
        LOGGER.info("Formulation: %s", formulation)

        missing = set(design_variables) - self.inputs
        if missing:
            raise ValueError(
                "{}: some design variables are not "
                "the inputs of any discipline: {}".format(scn_name, list(missing))
            )

        missing = (
            set(disciplines)
            - set(list(self.disciplines.keys()))
            - set(list(self.scenarios))
        )
        if missing:
            raise ValueError(
                "{}: some disciplines don't exist: {}".format(
                    scn_name, str(list(missing))
                )
            )

        missing = set(constraints) - self.outputs
        if missing:
            raise ValueError(
                "Some constraints of {} are not outputs of any discipline: {}".format(
                    scn_name, list(missing)
                )
            )

        missing = set(objectives) - self.outputs
        if missing:
            raise ValueError(
                "Some objectives of {} are not "
                "outputs of any discipline: {}".format(scn_name, list(missing))
            )
        if not objectives:
            raise ValueError(f"No objectives of {scn_name} are defined")

        if formulation not in get_available_formulations():
            raise ValueError(
                "Unknown formulation '{}'; use one of: {}".format(
                    formulation, get_available_formulations()
                )
            )


class StudyAnalysis:
    """A MDO study analysis from an Excel specification.

    Generate a N2 (equivalent to the Design Structure Matrix) diagram,
    showing the couplings between the disciplines,
    and a XDSM (Extended Design Structure Matrix),
    showing the MDO process,
    from an Excel specification of the inputs, outputs, design variables,
    objectives and constraints.

    The input Excel files contains one sheet per discipline.
    The name of the sheet shall have the discipline name.
    The sheet shall have at least two columns,
    one for inputs and one for outputs,
    with the following format:

    .. table:: Disc1

        +--------+---------+
        | Inputs | Outputs |
        +========+=========+
        |  in1   |  out1   |
        +--------+---------+
        |  in2   |  out2   |
        +--------+---------+

    Empty lines are ignored.

    The scenarios (at least one, or multiple for distributed formulations)
    must appear in a Excel sheet name starting by "Scenario".

    The sheet shall have the following columns,
    with some constraints :

    - All of them are mandatory, even if empty for the constraints.
    - The order does not matter.
    - One and only one formulation must be declared.
    - At least one objective must be provided, and one design variable.

    .. table:: Scenario1

        +----------------+--------------------+----------------+----------------+----------------+----------------+----------------+
        |Design variables| Objective function |  Constraints   |  Disciplines   |  Formulation   |  Options       | Options values |
        +================+====================+================+================+================+================+================+
        |      in1       |       out1         |     out2       |     Disc1      |     MDF        |  tolerance     |     0.1        |
        +----------------+--------------------+----------------+----------------+----------------+----------------+----------------+
        |                |                    |                |     Disc2      |                | main_mda_name  |   MDAJacobi    |
        +----------------+--------------------+----------------+----------------+----------------+----------------+----------------+

    All the objective functions and constraints must be outputs of a discipline,
    not necessarily the one of the current sheet.
    All the design variables must be inputs of a discipline,
    not necessarily the one of the current sheet.

    The columns 'Options' and 'Options values' are used to pass the formulation options.
    Note that for string type 'Option values',
    the value can be written with or without the "" characters.

    To use multi level MDO formulations,
    create multiple scenarios,
    and add the name of the sub scenarios
    in the list of disciplines of the main (system) scenario.

    An arbitrary number of levels can be generated this way
    (three, four levels etc formulations).
    """  # noqa: B950

    xls_study_path: str
    """The path of the Excel file."""

    study: XLSStudyParser
    """The XLSStudyParser instance built from the Excel file."""

    disciplines_descr: dict[str, MDODiscipline]
    """The descriptions of the disciplines
    (including sub-scenario) parsed in the Excel file."""

    scenarios_descr: dict[str, dict[str, str | list[str]]]
    """The descriptions of the scenarios parsed in the Excel file."""

    disciplines: dict[str, MDODiscipline]
    """The disciplines."""

    scenarios: dict[str, MDOScenario]
    """The scenarios."""

    AVAILABLE_DISTRIBUTED_FORMULATIONS = ("BiLevel", "BLISS98B")

    def __init__(self, xls_study_path: str) -> None:
        """Initialize the study from the Excel specification.

        Args:
            xls_study_path: The path to the Excel file describing the study.
        """
        self.xls_study_path = xls_study_path
        self.study = XLSStudyParser(self.xls_study_path)
        self.disciplines_descr = self.study.disciplines
        self.scenarios_descr = self.study.scenarios
        self.disciplines = dict()
        self.scenarios = dict()
        self.main_scenario = None
        self._create_scenarios()

    def generate_n2(
        self,
        file_path: str = "n2.pdf",
        show_data_names: bool = True,
        save: bool = True,
        show: bool = False,
        fig_size: tuple[float, float] = (15, 10),
    ) -> None:
        """Generate a N2 plot for the disciplines list.

        Args:
            file_path: The file path of the figure.
            show_data_names: If true,
                the names of the coupling data is shown;
                otherwise,
                circles are drawn,
                which size depends on the number of coupling names.
            save: If True, save the figure to file_path.
            show: If True, show the plot.
            fig_size: The size of the figure.
        """
        generate_n2_plot(
            list(self.disciplines.values()),
            file_path,
            show_data_names,
            save,
            show,
            fig_size,
        )

    @staticmethod
    def _create_scenario(
        disciplines: Iterable[MDODiscipline],
        scenario_description: Mapping[str, Iterable[str]],
    ) -> MDOScenario:
        """Create a MDO scenario.

        Args:
            disciplines: The disciplines.
            scenario_description: The description of the scenario.

        Returns:
            A MDO scenario.
        """
        design_space = create_design_space()
        coupling_variables = set(MDOCouplingStructure(disciplines).all_couplings)
        design_variables = set(scenario_description[XLSStudyParser.DESIGN_VARIABLES])
        for name in sorted(coupling_variables | design_variables):
            design_space.add_variable(name, size=1)

        option_names = scenario_description[XLSStudyParser.OPTIONS]
        options = {}
        if option_names is not None:
            option_values = scenario_description[XLSStudyParser.OPTIONS_VALUES]
            for option_name, option_value in zip(option_names, option_values):
                if isinstance(option_value, str):
                    try:
                        option_value = literal_eval(option_value)
                    except ValueError:
                        pass
                else:
                    pass

                options[option_name] = option_value

        scenario = create_scenario(
            disciplines,
            scenario_description[XLSStudyParser.FORMULATION],
            scenario_description[XLSStudyParser.OBJECTIVE_FUNCTION],
            design_space,
            **options,
        )
        for constraint_name in scenario_description[XLSStudyParser.CONSTRAINTS]:
            scenario.add_constraint(constraint_name)

        return scenario

    def _get_disciplines_instances(
        self, scn: Mapping[str, Iterable[str]]
    ) -> list[MDODiscipline]:
        """Get the instances of the disciplines from a scenario.

        Args:
            scn: The description of the scenario.

        Returns:
            The instances of the disciplines of the scenario.
        """
        discs = []
        for disc_name in scn[XLSStudyParser.DISCIPLINES]:
            disc_inst = self.disciplines_descr.get(disc_name)
            if disc_inst is None:  # not a discipline, so maybe a scenario
                disc_inst = self.scenarios.get(disc_name)
                if disc_inst is None:
                    return None
            discs.append(disc_inst)
        return discs

    def _create_scenarios(self) -> None:
        """Create the main scenario, eventually including sub scenarios.

        Raises:
            ValueError: If crossed dependencies exist between scenarios.
        """
        n_scn = len(self.scenarios_descr)
        i = 0

        temp_discs = {}
        while len(self.scenarios) != n_scn and i <= n_scn:
            i += 1
            for name, scn in self.scenarios_descr.items():
                discs = self._get_disciplines_instances(scn)
                if discs is not None:  # All depdendencies resolved
                    for disc in discs:
                        if not disc.is_scenario():
                            temp_discs[disc.name] = disc

                    scenario = self._create_scenario(discs, scn)
                    self.scenarios[name] = scenario
                    # The last scenario created is the one
                    # with the most dependencies
                    # so the main one
                    self.main_scenario = scenario
        # At each while iteration at least 1 scenario must be resolved
        # otherwise this means there is a cross dependency between
        # scenarios
        if len(self.scenarios) != n_scn:
            raise ValueError(
                "Scenarios dependencies cannot be resolved, "
                "check for cycling dependencies between scenarios!"
            )

        self.disciplines = dict()
        # Preserves the order of disciplines in the original Excel file
        # Important for the N2 generation
        for disc_name in self.disciplines_descr:
            self.disciplines[disc_name] = temp_discs[disc_name]

    def generate_xdsm(
        self,
        output_dir: str,
        latex_output: bool = False,
        open_browser: bool = False,
    ) -> MDOScenario:
        """Create an xdsm.json file from the current scenario.

        Args:
            output_dir: The directory where the XDSM html files are generated.
            latex_output: If True, build the .tex, .tikz and .pdf files.
            open_browser: If True, open in a web browser.

        Returns:
            The MDOScenario that contains the DesignSpace, the
            formulation, but the disciplines have only correct
            input and output grammars but no _run methods so that can't be executed
        """
        LOGGER.info("Generated the following Scenario:")
        LOGGER.info("%s", self.main_scenario)
        LOGGER.info("%s", self.main_scenario.formulation.opt_problem)
        self.main_scenario.xdsmize(
            outdir=output_dir, latex_output=latex_output, open_browser=open_browser
        )
        return self.main_scenario
