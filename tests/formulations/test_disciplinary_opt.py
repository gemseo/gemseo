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
#                       initial documentation
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import unittest

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt import DisciplinaryOpt
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiStructure


class TestDisciplinaryOpt(unittest.TestCase):
    """"""

    def test_multiple_disc(self):
        """"""
        ds = SobieskiProblem().design_space
        dopt = DisciplinaryOpt([SobieskiStructure(), SobieskiMission()], "y_4", ds)
        dopt.get_expected_dataflow()
        dopt.get_expected_workflow()

    def test_init(self):
        """"""
        sm = SobieskiMission()
        ds = SobieskiProblem().design_space
        dopt = DisciplinaryOpt([sm], "y_4", ds)
        assert dopt.get_expected_dataflow() == []
        assert dopt.get_expected_workflow().sequences[0].discipline == sm
        assert len(dopt.get_expected_workflow().sequences) == 1


def test_grammar_type():
    """Check that the grammar type is correctly used."""
    discipline = AnalyticDiscipline({"y": "x"})
    design_space = DesignSpace()
    design_space.add_variable("x")
    grammar_type = discipline.SIMPLE_GRAMMAR_TYPE
    formulation = DisciplinaryOpt(
        [discipline] * 2, "y", design_space, grammar_type=grammar_type
    )
    assert formulation.chain.grammar_type == grammar_type
