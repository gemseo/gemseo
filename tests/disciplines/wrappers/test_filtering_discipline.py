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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest

from gemseo import create_discipline
from gemseo.disciplines.wrappers.filtering_discipline import FilteringDiscipline


@pytest.fixture
def discipline():
    expressions = {"y1": "x1+x2+x3", "y2": "-x1-x2-x3"}
    disc = create_discipline("AnalyticDiscipline", expressions=expressions, name="foo")
    disc.io.input_grammar.descriptions = {"x1": "This is x1.", "x2": "This is x2."}
    disc.io.output_grammar.descriptions = {"y1": "This is y1.", "y2": "This is y2."}
    disc.add_differentiated_inputs(["x1", "x2", "x3"])
    disc.add_differentiated_outputs(["y1", "y2"])
    return disc


def test_standard(discipline) -> None:
    fdisc = FilteringDiscipline(discipline)
    assert set(fdisc.io.input_grammar) == set(discipline.io.input_grammar)
    assert set(fdisc.io.output_grammar) == set(discipline.io.output_grammar)
    fdisc.execute()
    for name in ["x1", "x2", "x3", "y1", "y2"]:
        assert name in fdisc.io.data
    fdisc.linearize()
    for output_name in ["y1", "y2"]:
        assert output_name in fdisc.jac
        for input_name in ["x1", "x2", "x3"]:
            assert input_name in fdisc.jac[output_name]

    assert (
        fdisc.io.input_grammar.descriptions == discipline.io.input_grammar.descriptions
    )
    assert (
        fdisc.io.output_grammar.descriptions
        == discipline.io.output_grammar.descriptions
    )


def test_keep_in_keep_out(discipline) -> None:
    fdisc = FilteringDiscipline(discipline, input_names=["x1"], output_names=["y1"])
    assert set(fdisc.io.input_grammar) == {"x1"}
    assert set(fdisc.io.output_grammar) == {"y1"}
    fdisc.execute()
    for name in ["x2", "x3", "y2"]:
        assert name not in fdisc.io.data
    fdisc.linearize()
    assert "y2" not in fdisc.jac
    for input_name in ["x2", "x3"]:
        assert input_name not in fdisc.jac["y1"]

    assert fdisc.io.input_grammar.descriptions == {"x1": "This is x1."}
    assert fdisc.io.output_grammar.descriptions == {"y1": "This is y1."}


def test_remove_in_keep_out(discipline) -> None:
    fdisc = FilteringDiscipline(
        discipline, input_names=["x1"], output_names=["y1"], keep_in=False
    )
    assert set(fdisc.io.input_grammar) == {"x2", "x3"}
    assert set(fdisc.io.output_grammar) == {"y1"}
    fdisc.execute()
    for name in ["x1", "y2"]:
        assert name not in fdisc.io.data
    fdisc.linearize()
    assert "y2" not in fdisc.jac
    assert "x1" not in fdisc.jac["y1"]


def test_keep_in_remove_out(discipline) -> None:
    fdisc = FilteringDiscipline(
        discipline, input_names=["x1"], output_names=["y1"], keep_out=False
    )
    assert set(fdisc.io.input_grammar) == {"x1"}
    assert set(fdisc.io.output_grammar) == {"y2"}
    fdisc.execute()
    for name in ["x2", "x3", "y1"]:
        assert name not in fdisc.io.data
    fdisc.linearize()
    assert "y1" not in fdisc.jac
    for input_name in ["x2", "x3"]:
        assert input_name not in fdisc.jac["y2"]
