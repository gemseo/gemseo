# -*- coding: utf-8 -*-
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
"""
Generate a N2 and XDSM from an Excel description of the MDO problem
*******************************************************************
"""
from __future__ import absolute_import, division, unicode_literals

from ast import literal_eval

from future import standard_library
from pandas import read_excel
from six import string_types

from gemseo import LOGGER
from gemseo.api import (
    create_design_space,
    create_scenario,
    generate_n2_plot,
    get_available_formulations,
)
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline

standard_library.install_aliases()


class XLSStudyParser(object):
    """
    Parse the input excel files that describe the GEMSEO study.

    The Excel file must contain one sheet per discipline.
    The name of the sheet is the discipline name.
    The sheet has at least two columns, one for inputs and one for outputs,
    with the following format:

    +--------+---------+----------------+--------------------+----------------+
    | Inputs | Outputs |Design variables| Objective function |  Constraints   |
    +========+=========+================+====================+================+
    |  in1   |  out1   |      in1       |       out1         |     out2       |
    +--------+---------+----------------+--------------------+----------------+
    |  in2   |  out2   |                |                    |                |
    +--------+---------+----------------+--------------------+----------------+

    Empty lines are ignored.
    All Objective functions and constraints must be outputs of a discipline,
    not necessarily the one of the current sheet
    All Design variables must be inputs of a discipline, not necessarily
    the one of the current sheet

    """

    SCENARIO_PREFIX = "Scenario"
    DISCIPLINE = "Discipline"
    DISCIPLINES = "Disciplines"
    OBJECTIVE_FUNCTION = "Objective function"
    CONSTRAINTS = "Constraints"
    DESIGN_VARIABLES = "Design variables"
    FORMULATION = "Formulation"
    OPTIONS = "Options"
    OPTIONS_VALUES = "Options values"

    def __init__(self, xls_study_path):
        """
        Initializes the study from the excel specification

        :param xls_study_path: path to the excel file describing the study
        """

        self.xls_study_path = xls_study_path
        try:
            self.frames = read_excel(xls_study_path, sheet_name=None, engine="openpyxl")
        except IOError:
            LOGGER.error("Failed to open study file !")
            raise

        LOGGER.info("Detected the following disciplines: %s", list(self.frames.keys()))
        self.disciplines = {}
        self.scenarios = {}
        self.inputs = []
        self.outputs = []

        self._init_disciplines()
        self._get_opt_pb_descr()

        if not self.scenarios:
            raise ValueError("Found no scenario in the xls file !")

    def _init_disciplines(self):
        """ Initializes disciplines. """
        for disc_name, frame in self.frames.items():
            if disc_name.startswith(self.SCENARIO_PREFIX):
                continue
            LOGGER.info("Parsing discipline %s", disc_name)

            try:
                inputs = self._get_frame_series_values(frame, "Inputs")
            except ValueError:
                raise ValueError(
                    "Discipline "
                    + str(disc_name)
                    + "'s sheet must have an Inputs column !"
                )
            self.inputs += inputs
            try:
                outputs = self._get_frame_series_values(frame, "Outputs")
            except ValueError:
                raise ValueError(
                    "Discipline "
                    + str(disc_name)
                    + "'s sheet must have an Outputs column !"
                )

            if not outputs:
                raise ValueError("Discipline " + str(disc_name) + " has no Outputs")
            self.outputs += outputs
            disc = MDODiscipline(disc_name)
            disc.input_grammar.initialize_from_data_names(inputs)
            disc.output_grammar.initialize_from_data_names(outputs)
            LOGGER.info("Inputs : %s", inputs)
            LOGGER.info("Outputs : %s", outputs)

            self.disciplines[disc_name] = disc

        self.inputs = set(self.inputs)
        self.outputs = set(self.outputs)

    @staticmethod
    def _get_frame_series_values(frame, series_name, return_none=False):
        """
        Gets the data list of a named column
        Removes empty data
        :param frame: the pandas frame of the sheet
        :param series_name: name of the series
        :param return_none: if the series does not exists, returns None
             instead of raising a ValueError
        """
        series = frame.get(series_name)
        if series is None:
            if return_none:
                return None
            raise ValueError("The sheet has no series named " + str(series_name))
        # Remove empty data
        # pylint: disable=comparison-with-itself
        return list([val for val in series.tolist() if val == val])

    def _get_opt_pb_descr(self):
        """
        Initilalize the objective function, constraints and design_variables
        """
        self.scenarios = {}

        for frame_name, frame in self.frames.items():
            if not frame_name.startswith(self.SCENARIO_PREFIX):
                continue
            LOGGER.info("Detected scenario in sheet: %s", frame_name)

            try:
                disciplines = self._get_frame_series_values(frame, self.DISCIPLINES)
            except ValueError:
                raise ValueError(
                    "Scenario "
                    + str(frame_name)
                    + " has no "
                    + self.DISCIPLINES
                    + " column !"
                )
            try:
                design_variables = self._get_frame_series_values(
                    frame, self.DESIGN_VARIABLES
                )
            except ValueError:
                raise ValueError(
                    "Scenario "
                    + str(frame_name)
                    + " has no "
                    + self.DESIGN_VARIABLES
                    + " column !"
                )
            try:
                objectives = self._get_frame_series_values(
                    frame, self.OBJECTIVE_FUNCTION
                )
            except ValueError:
                raise ValueError(
                    "Scenario "
                    + str(frame_name)
                    + " has no "
                    + self.OBJECTIVE_FUNCTION
                    + " column !"
                )
            try:
                constraints = self._get_frame_series_values(frame, self.CONSTRAINTS)
            except ValueError:
                raise ValueError(
                    "Scenario "
                    + str(frame_name)
                    + " has no "
                    + self.CONSTRAINTS
                    + " column !"
                )

            try:
                formulation = self._get_frame_series_values(frame, self.FORMULATION)
            except ValueError:
                raise ValueError(
                    "Scenario "
                    + str(frame_name)
                    + " has no "
                    + self.FORMULATION
                    + " column !"
                )

            options = self._get_frame_series_values(frame, self.OPTIONS, True)
            options_values = self._get_frame_series_values(
                frame, self.OPTIONS_VALUES, True
            )

            if len(formulation) != 1:
                raise ValueError(
                    "Scenario "
                    + str(frame_name)
                    + " must have 1 "
                    + self.FORMULATION
                    + " value !"
                )

            if options is not None:
                if len(options) != len(options_values):
                    raise ValueError(
                        "Options "
                        + str(options)
                        + " and Options values "
                        + str(options_values)
                        + " must have the same length!"
                    )

            formulation = formulation[0]

            scn = {}
            scn[self.DISCIPLINES] = disciplines
            scn[self.OBJECTIVE_FUNCTION] = objectives
            scn[self.CONSTRAINTS] = constraints
            scn[self.DESIGN_VARIABLES] = design_variables
            scn[self.FORMULATION] = formulation
            scn[self.OPTIONS] = options
            scn[self.OPTIONS_VALUES] = options_values

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
        objectives,
        constraints,
        disciplines,
        design_variables,
        formulation,
        scn_name,
    ):
        """
        Checks the optimization problem consistency.
        Raises errors if needed

        :param objectives: list of objectives
        :param constraints: list of constraints
        :param disciplines: list of MDODisciplines
        :param design_variables: list of design varaibles
        :param formulation : mdo formulation name
        :param scn_name: name of the scenario

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
                scn_name + " : some design variables are "
                "not the inputs of any discipline :" + str(list(missing))
            )

        missing = (
            set(disciplines)
            - set(list(self.disciplines.keys()))
            - set(list(self.scenarios))
        )
        if missing:
            raise ValueError(
                scn_name + " : some disciplines dont exist :" + str(list(missing))
            )

        missing = set(constraints) - self.outputs
        if missing:
            raise ValueError(
                scn_name + " : some constraints are not "
                "the outputs of any discipline :" + str(list(missing))
            )

        missing = set(objectives) - self.outputs
        if missing:
            raise ValueError(
                scn_name + " : some objectives are not "
                "the outputs of any discipline :" + str(list(missing))
            )
        if not objectives:
            raise ValueError(scn_name + " : no objectives are defined!")

        if formulation not in get_available_formulations():
            raise ValueError(
                "Unknown formulation "
                + str(formulation)
                + " Use one of "
                + str(get_available_formulations())
            )


class StudyAnalysis(object):
    """
    Generate a N2 (equivalent to the Design Structure Matrix) diagram,
    showing the couplings between discipline and
    XDSM, (Extended Design Structure Matrix), showing the MDO process,
    from a Excel specification of the inputs, outputs, design variables,
    objectives and constraints.

    The input excel files contains one sheet per discipline.
    The name of the sheet is the discipline name.
    The sheet has at least two columns, one for inputs and one for outputs,
    with the following format:


    +--------+---------+
    | Inputs | Outputs |
    +========+=========+
    |  in1   |  out1   |
    +--------+---------+
    |  in2   |  out2   |
    +--------+---------+

    [Disc1]

    Empty lines are ignored.


    The scenarios (at least one, or multiple for distributed formulations)
    must appear in a Excel sheet name starting by "Scenario".

    The sheet has the following columns, with some constraints :
    All of them are mandatory, even if empty for the Constraints
    The order may be any
    1 and only 1 formulation must be declared
    At least 1 objective must be provided, and 1 design variable

    +----------------+--------------------+----------------+----------------+----------------+----------------+----------------+
    |Design variables| Objective function |  Constraints   |  Disciplines   |  Formulation   |  Options       | Options values |
    +================+====================+================+================+================+================+================+
    |      in1       |       out1         |     out2       |     Disc1      |     MDF        |  tolerance     |     0.1        |
    +----------------+--------------------+----------------+----------------+----------------+----------------+----------------+
    |                |                    |                |     Disc2      |                |                |                |
    +----------------+--------------------+----------------+----------------+----------------+----------------+----------------+

    [Scenario1]

    All Objective functions and constraints must be outputs of a discipline,
    not necessarily the one of the current sheet.
    All Design variables must be inputs of a discipline, not necessarily
    the one of the current sheet.

    The Options and Options values columns are used to pass
    the formulation options

    To use multi level MDO formulations, create multiple scenarios,
    and add the name of the sub scenarios
    in the list of disciplines of the main (system) scenario.

    An arbitrary number of levels can be generated this way
    (three, four levels etc formulations).
    """

    AVAILABLE_DISTRIBUTED_FORMULATIONS = ("BiLevel", "BLISS98B")

    def __init__(self, xls_study_path):
        """
        Initializes the study from the excel specification

        :param xls_study_path: path to the excel file describing the study
        """

        self.xls_study_path = xls_study_path
        self.study = XLSStudyParser(self.xls_study_path)
        self.disciplines_descr = self.study.disciplines
        self.scenarios_descr = self.study.scenarios
        self.disciplines = {}
        self.scenarios = {}
        self.main_scenario = None
        self._create_scenarios()

    def generate_n2(
        self,
        file_path="n2.pdf",
        show_data_names=True,
        save=True,
        show=False,
        figsize=(15, 10),
    ):
        """
        Generate a N2 plot for the disciplines list.

        :param file_path: File path of the figure.
        :type file_path: str
        :param show_data_names: If true, the names of the
            coupling data is shown
            otherwise, circles are drawn, which size depend on the
            number of coupling names.
        :type show_data_names: bool
        :param save: If True, saved the figure to file_path.
        :type save: bool
        :param show: If True, shows the plot.
        :type show: bool
        :param figsize: Size of the figure.
        :type figsize: tuple(float)
        """
        generate_n2_plot(
            list(self.disciplines.values()),
            file_path,
            show_data_names,
            save,
            show,
            figsize,
        )

    @staticmethod
    def _create_scenario(disciplines, scenario_descr):
        """
        Create a MDO scenario

        :param disciplines: list of MDODisciplines
        :param scenario_descr: description dict of the scenario
        :returns: the MDOScenario
        """
        coupl_struct = MDOCouplingStructure(disciplines)
        couplings = coupl_struct.get_all_couplings()
        design_space = create_design_space()
        scn_dv = scenario_descr[XLSStudyParser.DESIGN_VARIABLES]
        for var in set(scn_dv) | set(couplings):
            design_space.add_variable(var, size=1)

        options = scenario_descr[XLSStudyParser.OPTIONS]
        options_dict = {}
        if options is not None:
            options_values = scenario_descr[XLSStudyParser.OPTIONS_VALUES]
            for opt, val in zip(options, options_values):
                if isinstance(val, string_types):
                    try:
                        val = literal_eval(val)
                    except ValueError as err:
                        LOGGER.error(err)
                        raise ValueError(
                            "Failed to parse option " + str(opt) + " value :" + str(val)
                        )
                else:
                    pass
                options_dict[opt] = val

        scenario = create_scenario(
            disciplines,
            scenario_descr[XLSStudyParser.FORMULATION],
            scenario_descr[XLSStudyParser.OBJECTIVE_FUNCTION],
            design_space,
            **options_dict
        )
        for cstr in scenario_descr[XLSStudyParser.CONSTRAINTS]:
            scenario.add_constraint(cstr)
        return scenario

    def _get_disciplines_instances(self, scn):
        """
        Returns instances of the disciplines of the scenario,
        or None if not all available
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

    def _create_scenarios(self):
        """
        Create the main scenario, eventually including sub scenarios
        """
        n_scn = len(self.scenarios_descr)
        i = 0

        while len(self.scenarios) != n_scn and i <= n_scn:
            i += 1
            for name, scn in self.scenarios_descr.items():
                discs = self._get_disciplines_instances(scn)
                if discs is not None:  # All depdendencies resolved
                    for disc in discs:
                        if not disc.is_scenario():
                            self.disciplines[disc.name] = disc

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
                "Scenarios dependencies cannot be resolved,"
                " check for cycling dependencies "
                "between scenarios!"
            )

    def generate_xdsm(self, output_dir, latex_output=False, open_browser=False):
        """
        Creates an xdsm.json file from the current scenario.

        :param output_dir: the directory where XDSM html files are generated
        :param latex_output: build .tex, .tikz and .pdf file
        :returns: the MDOScenario, that contains the DesignSpace, the
            formulation, but the disciplines have only correct
            input and output grammars but no _run methods so that can't be executed
        """
        LOGGER.info("Generated the following Scenario:")
        self.main_scenario.log_me()
        self.main_scenario.formulation.opt_problem.log_me()
        self.main_scenario.xdsmize(
            outdir=output_dir, latex_output=latex_output, open_browser=open_browser
        )
        return self.main_scenario
