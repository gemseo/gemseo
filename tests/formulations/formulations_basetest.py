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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import unittest

from gemseo.core.discipline import MDODiscipline
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure


class FakeDiscipline(MDODiscipline):
    """"""

    def _instantiate_grammars(
        self,
        input_grammar_file,
        output_grammar_file,
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,
    ):
        """

        :param input_grammar_file: param output_grammar_file:
        :param grammar_type: Default value = MDODiscipline.JSON_GRAMMAR_TYPE)
        :param output_grammar_file:

        """
        self.input_grammar = JSONGrammar("inputs")
        self.input_grammar.update_from_data({self.name + "_x": 0.0})
        self.output_grammar = JSONGrammar("outputs")
        self.output_grammar.update_from_data({self.name + "_y": 1.0})


class FormulationsBaseTest(unittest.TestCase):
    """"""

    DV_NAMES = ["x_1", "x_2", "x_3", "x_shared"]

    def build_mdo_scenario(self, formulation="MDF", dtype="complex128", **options):
        """

        :param formulation: Default value = 'MDF')
        :param dtype: Default value = "complex128")

        """
        disciplines = [
            SobieskiStructure(dtype),
            SobieskiPropulsion(dtype),
            SobieskiAerodynamics(dtype),
            SobieskiMission(dtype),
        ]
        design_space = SobieskiProblem().design_space
        return MDOScenario(
            disciplines,
            formulation=formulation,
            objective_name="y_4",
            design_space=design_space,
            maximize_objective=True,
            **options,
        )
