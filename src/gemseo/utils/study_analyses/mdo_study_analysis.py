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
"""An MDO study analysis generating N2 and XDSM from an Excel specification."""

from __future__ import annotations

import contextlib
import logging
from ast import literal_eval
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo import create_design_space
from gemseo import create_scenario
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.utils.study_analyses.coupling_study_analysis import CouplingStudyAnalysis
from gemseo.utils.study_analyses.xls_study_parser import XLSStudyParser

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from pathlib import Path

    from gemseo.core.discipline import MDODiscipline
    from gemseo.core.mdo_scenario import MDOScenario
    from gemseo.utils.xdsm import XDSM

LOGGER = logging.getLogger(__name__)


class MDOStudyAnalysis(CouplingStudyAnalysis):
    """An MDO study analysis from an Excel specification.

    Based on an Excel file defining
    disciplines
    in terms of input and output names
    and
    an MDO problem
    in terms of names of design variables, objectives, constraints and formulation,
    this analysis generates an N2 (equivalent to the Design Structure Matrix) diagram,
    showing the couplings between the disciplines,
    and an XDSM (Extended Design Structure Matrix),
    showing the MDO process.

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

    The scenarios (at least one, or multiple for distributed formulations)
    must appear in a Excel sheet with a name starting by ``Scenario``.

    The sheet shall have the following columns:

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

    _HAS_SCENARIO: ClassVar[bool] = True

    main_scenario: MDOScenario
    """The main scenario."""

    scenarios: dict[str, MDOScenario]
    """The sub-scenarios and the main scenario in the last position."""

    def __init__(self, xls_study_path: str | Path) -> None:  # noqa: D107
        super().__init__(xls_study_path)
        self.scenarios = {}
        self._create_scenarios()

    @staticmethod
    def _create_scenario(
        disciplines: Iterable[MDODiscipline],
        scenario_description: Mapping[str, Iterable[str]],
    ) -> MDOScenario:
        """Create an MDO scenario.

        Args:
            disciplines: The disciplines.
            scenario_description: The description of the scenario.

        Returns:
            An MDO scenario.
        """
        design_space = create_design_space()
        coupling_variables = set(MDOCouplingStructure(disciplines).all_couplings)
        design_variables = set(scenario_description[XLSStudyParser.DESIGN_VARIABLES])
        for name in sorted(coupling_variables | design_variables):
            design_space.add_variable(name)

        option_names = scenario_description[XLSStudyParser.OPTIONS]
        options = {}
        if option_names is not None:
            option_values = scenario_description[XLSStudyParser.OPTION_VALUES]
            for option_name, option_value in zip(option_names, option_values):
                if isinstance(option_value, str):
                    with contextlib.suppress(ValueError):
                        option_value = literal_eval(option_value)

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
        self, scenario_description: Mapping[str, Iterable[str]]
    ) -> list[MDODiscipline]:
        """Return the disciplines from a scenario.

        Args:
            scenario_description: The description of the scenario.

        Returns:
            The disciplines of the scenario.
        """
        disciplines = []
        for discipline_name in scenario_description[XLSStudyParser.DISCIPLINES]:
            discipline = self.study.disciplines.get(discipline_name)
            if discipline is None:
                # Not a discipline.
                discipline = self.scenarios.get(discipline_name)
                if discipline is None:
                    # Not a scenario.
                    return []
            disciplines.append(discipline)
        return disciplines

    def _create_scenarios(self) -> None:
        """Create the main scenario, eventually including sub-scenarios.

        Raises:
            ValueError: If crossed dependencies exist between scenarios.
        """
        n_study_scenarios = len(self.study.scenarios)
        non_scenario_disciplines = {}
        i = 0
        while len(self.scenarios) != n_study_scenarios and i <= n_study_scenarios:
            i += 1
            for name, scenario_description in self.study.scenarios.items():
                disciplines = self._get_disciplines_instances(scenario_description)
                if disciplines:  # All dependencies resolved
                    for discipline in disciplines:
                        if not discipline.is_scenario():
                            non_scenario_disciplines[discipline.name] = discipline

                    scenario = self._create_scenario(disciplines, scenario_description)
                    self.scenarios[name] = scenario
                    # The last scenario created is the one
                    # with the most dependencies
                    # so the main one
                    self.main_scenario = scenario

        # At each while iteration at least 1 scenario must be resolved
        # otherwise this means there is a cross dependency between
        # scenarios
        if len(self.scenarios) != n_study_scenarios:
            raise ValueError(
                "Scenarios dependencies cannot be resolved, "
                "check for cycling dependencies between scenarios."
            )

        self.disciplines = {}
        # Preserves the order of disciplines in the original Excel file
        # Important for the N2 generation
        for disc_name in self.study.disciplines:
            self.disciplines[disc_name] = non_scenario_disciplines[disc_name]

    def generate_xdsm(
        self,
        directory_path: str | Path = ".",
        save_pdf: bool = False,
        show_html: bool = False,
    ) -> XDSM:
        """Create an XDSM diagram of the :attr:`.main_scenario`.

        Args:
            directory_path: The path of the directory to save the files.
            save_pdf: Whether to save the XDSM as a PDF file.
            show_html: Whether to open the web browser and display the XDSM.

        Returns:
            The XDSM diagram of the :attr:`.main_scenario`.
        """
        LOGGER.info("Generated the following Scenario:")
        LOGGER.info("%s", self.main_scenario)
        LOGGER.info("%s", self.main_scenario.formulation.opt_problem)
        return self.main_scenario.xdsmize(
            directory_path=directory_path, save_pdf=save_pdf, show_html=show_html
        )
