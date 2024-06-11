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

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt import DisciplinaryOpt
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure


def test_multiple_disc() -> None:
    """"""
    ds = SobieskiDesignSpace()
    dopt = DisciplinaryOpt([SobieskiStructure(), SobieskiMission()], "y_4", ds)
    dopt.get_expected_dataflow()
    dopt.get_expected_workflow()


def test_init() -> None:
    """"""
    sm = SobieskiMission()
    ds = SobieskiDesignSpace()
    dopt = DisciplinaryOpt([sm], "y_4", ds)
    assert dopt.get_expected_dataflow() == []
    assert dopt.get_expected_workflow().sequences[0].discipline == sm
    assert len(dopt.get_expected_workflow().sequences) == 1


def test_grammar_type() -> None:
    """Check that the grammar type is correctly used."""
    discipline = AnalyticDiscipline({"y": "x"})
    design_space = DesignSpace()
    design_space.add_variable("x")
    grammar_type = discipline.GrammarType.SIMPLE
    formulation = DisciplinaryOpt(
        [discipline] * 2, "y", design_space, grammar_type=grammar_type
    )
    assert formulation.chain.grammar_type == grammar_type


@pytest.mark.parametrize(
    ("options", "expected_jac"),
    [
        ({}, array([2.0])),
        ({"differentiated_input_names_substitute": ["a"]}, array([2.0])),
        ({"differentiated_input_names_substitute": ["b"]}, array([3.0])),
        ({"differentiated_input_names_substitute": ["a", "b"]}, array([2.0, 3.0])),
        ({"differentiated_input_names_substitute": ["b", "a"]}, array([3.0, 2.0])),
    ],
)
def test_jac_wrt_dv_or_non_dv(options, expected_jac):
    """Check the Jacobian wrt design or non-design input variables."""
    discipline = AnalyticDiscipline({"f": "2*a+3*b", "c": "2*a+3*b", "o": "2*a+3*b"})

    design_space = DesignSpace()
    design_space.add_variable("a")

    formulation = DisciplinaryOpt([discipline], "f", design_space, **options)
    formulation.add_constraint("c")
    formulation.add_observable("o")
    problem = formulation.optimization_problem

    for function in [problem.objective, problem.constraints[0], problem.observables[0]]:
        assert_equal(function.evaluate(array([1])), array([2.0]))
        assert_equal(function.jac(array([1])), expected_jac)
