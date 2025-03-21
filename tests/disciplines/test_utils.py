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
from __future__ import annotations

import logging
import re

import pytest

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.discipline import DummyDiscipline
from gemseo.utils.discipline import check_disciplines_consistency
from gemseo.utils.discipline import get_all_inputs
from gemseo.utils.discipline import get_all_outputs


class MyDiscipline(DummyDiscipline):
    def __init__(self, input_name: str, output_name: str, discipline_name: str) -> None:
        super().__init__(discipline_name)
        self.io.input_grammar.update_from_names([input_name])
        self.io.output_grammar.update_from_names([output_name])


@pytest.fixture(scope="module")
def consistent_disciplines() -> tuple[MyDiscipline, MyDiscipline]:
    """Two consistent disciplines."""
    return MyDiscipline("a", "y", "Foo"), MyDiscipline("b", "z", "Bar")


@pytest.fixture(scope="module")
def inconsistent_disciplines() -> tuple[MyDiscipline, MyDiscipline]:
    """Two inconsistent disciplines."""
    return MyDiscipline("a", "y", "Foo"), MyDiscipline("b", "y", "Bar")


@pytest.fixture(scope="module")
def disciplines_and_scenario() -> list[MyDiscipline]:
    """Disciplines with a scenario."""
    disciplines = [
        AnalyticDiscipline({"y1": "x1"}, name="f1"),
        AnalyticDiscipline({"y2": "x2"}, name="f2"),
    ]
    sub_disciplines = [
        AnalyticDiscipline({"ya": "xa"}, name="fa"),
        AnalyticDiscipline({"yb": "xb"}, name="fb"),
    ]
    design_space = DesignSpace()
    design_space.add_variable("xa")
    scenario = MDOScenario(
        sub_disciplines, "ya", design_space, formulation_name="DisciplinaryOpt"
    )
    return [*disciplines, scenario]


@pytest.mark.parametrize(
    ("skip_scenarios", "expected"),
    [(True, ["x1", "x2"]), (False, ["x1", "x2", "xa", "xb"])],
)
def test_get_all_inputs(disciplines_and_scenario, skip_scenarios, expected) -> None:
    """Check get_all_inputs."""
    assert get_all_inputs(disciplines_and_scenario, skip_scenarios) == expected


@pytest.mark.parametrize(
    ("skip_scenarios", "expected"),
    [(True, ["y1", "y2"]), (False, ["y1", "y2", "ya", "yb"])],
)
def test_get_all_outputs(disciplines_and_scenario, skip_scenarios, expected) -> None:
    """Check get_all_outputs."""
    assert get_all_outputs(disciplines_and_scenario, skip_scenarios) == expected


@pytest.mark.parametrize("log_message", [False, True])
@pytest.mark.parametrize("raise_error", [False, True])
def test_check_disciplines_consistency(
    consistent_disciplines, log_message, raise_error
) -> None:
    """Test check_disciplines_consistency with consistent disciplines."""
    assert check_disciplines_consistency(
        consistent_disciplines, log_message, raise_error
    )


def test_check_disciplines_consistency_log(inconsistent_disciplines, caplog) -> None:
    """Test check_disciplines_consistency with inconsistent disciplines and log mode."""
    assert not check_disciplines_consistency(inconsistent_disciplines, True, False)
    record = caplog.record_tuples[0]
    assert record[1] == logging.WARNING
    assert (
        record[2] == "Two disciplines, among which Bar, compute the same outputs: {'y'}"
    )


def test_check_disciplines_consistency_error(inconsistent_disciplines, caplog) -> None:
    """Test check_disciplines_consistency with inconsistent disciplines and log mode."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Two disciplines, among which Bar, compute the same outputs: {'y'}"
        ),
    ):
        check_disciplines_consistency(inconsistent_disciplines, False, True)
