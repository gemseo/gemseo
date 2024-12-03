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
from typing import ClassVar

from gemseo.core.discipline import Discipline
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario


class FakeDiscipline(Discipline):
    """"""

    def __init__(self, name: str):
        super().__init__(name)
        self.io.input_grammar = JSONGrammar("inputs")
        self.io.input_grammar.update_from_data({self.name + "_x": 0.0})
        self.io.output_grammar = JSONGrammar("outputs")
        self.io.output_grammar.update_from_data({self.name + "_y": 1.0})


class FormulationsBaseTest(unittest.TestCase):
    """"""

    DV_NAMES: ClassVar[list[str]] = ["x_1", "x_2", "x_3", "x_shared"]

    def build_mdo_scenario(
        self,
        formulation_name="MDF",
        dtype="complex128",
        **options,
    ):
        """

        :param formulation_name: Default value = 'MDF')
        :param dtype: Default value = "complex128")

        """
        disciplines = [
            SobieskiStructure(dtype),
            SobieskiPropulsion(dtype),
            SobieskiAerodynamics(dtype),
            SobieskiMission(dtype),
        ]
        design_space = SobieskiDesignSpace()
        return MDOScenario(
            disciplines,
            "y_4",
            design_space,
            formulation_name=formulation_name,
            maximize_objective=True,
            **options,
        )
