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
#     Matthias De Lozzo
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import zeros
from numpy.testing import assert_equal

from gemseo.core.discipline import Discipline
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.disciplines.remapping import RemappingDiscipline
from gemseo.utils.discipline import DummyDiscipline
from gemseo.utils.pickle import from_pickle
from gemseo.utils.pickle import to_pickle

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


class NewDiscipline(Discipline):
    """A new discipline."""

    def __init__(self) -> None:
        super().__init__(name="foo")
        default_input_data = {
            "in_1": array([1.0]),
            "in_2": array([2.0, 3.0]),
            "in_3": array(["zero"]),
        }
        self.io.input_grammar.update_from_data(default_input_data)
        self.io.output_grammar.update_from_data({
            "out_1": array([2.0]),
            "out_2": array([1.0, 2.0]),
            "out_3": array(["zero plus one"]),
        })
        self.io.input_grammar.defaults = default_input_data

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        self.io.data["out_1"] = self.io.data["in_1"] + 1
        self.io.data["out_2"] = self.io.data["in_2"] - 1
        self.io.data["out_3"] = array([f"{self.io.data['in_3'][0]} plus one"])

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        self.jac = {
            "out_1": {
                "in_1": array([[1.0]]),
                "in_2": array([[1.0, 1.0]]),
                "in_3": zeros((1, 1)),
            },
            "out_2": {
                "in_1": array([[1.0], [1.0]]),
                "in_2": array([[1.0, 1.0], [1.0, 1.0]]),
                "in_3": zeros((2, 1)),
            },
            "out_3": {
                "in_1": zeros((1, 1)),
                "in_2": zeros((1, 2)),
                "in_3": zeros((1, 1)),
            },
        }


input_mapping = {
    "new_in_1": "in_1",
    "new_in_2": ("in_2", 0),
    "new_in_3": ("in_2", 1),
    "new_in_4": ("in_3"),
}
output_mapping = {"new_out_1": "out_1", "new_out_2": "out_2", "new_out_3": "out_3"}


@pytest.fixture(scope="module", params=[False, True])
def discipline(module_tmp_wd, request) -> Discipline:
    """A remapping discipline."""
    discipline = RemappingDiscipline(NewDiscipline(), input_mapping, output_mapping)
    if not request.param:
        # Use the original remapping discipline
        return discipline

    # Use the remapping discipline loaded from the disk, after serialization
    file_name = "discipline.pkl"
    to_pickle(discipline, file_name)
    return from_pickle(file_name)


def test_original_discipline(discipline) -> None:
    """Check the property original_discipline."""
    assert discipline.original_discipline == discipline._discipline


def test_with_discipline_wo_default_values() -> None:
    """Check that the wrapped discipline must have default input values."""
    with pytest.raises(
        ValueError,
        match=re.escape("The original discipline has no default input values."),
    ):
        RemappingDiscipline(DummyDiscipline(), {}, {})


def test_discipline_name(discipline) -> None:
    """Check that the discipline name is the name of the original discipline."""
    assert discipline.name == "foo"


def test_io_names(discipline) -> None:
    """Check the input and output names."""
    assert discipline.io.input_grammar.keys() == input_mapping.keys()
    assert discipline.io.output_grammar.keys() == output_mapping.keys()


def test_default_inputs(discipline) -> None:
    """Check the default inputs when missing in original discipline."""
    assert_equal(
        {
            "new_in_1": array([1.0]),
            "new_in_2": array([2.0]),
            "new_in_3": array([3.0]),
            "new_in_4": array(["zero"]),
        },
        discipline.io.input_grammar.defaults,
    )


def test_execute(discipline) -> None:
    """Check the execution of the discipline."""
    discipline.execute()
    assert_equal(discipline.io.data["new_out_1"], array([2.0]))
    assert_equal(discipline.io.data["new_out_2"], array([1.0, 2.0]))


def test_linearize_all(discipline) -> None:
    """Check the linearization of all the inputs/outputs of the discipline."""
    discipline.linearize(compute_all_jacobians=True)
    assert_equal(
        discipline.jac,
        {
            "new_out_1": {
                "new_in_1": array([[1.0]]),
                "new_in_2": array([[1.0]]),
                "new_in_3": array([[1.0]]),
                "new_in_4": zeros((1, 1)),
            },
            "new_out_2": {
                "new_in_1": array([[1.0], [1.0]]),
                "new_in_2": array([[1.0], [1.0]]),
                "new_in_3": array([[1.0], [1.0]]),
                "new_in_4": zeros((2, 1)),
            },
            "new_out_3": {
                "new_in_1": zeros((1, 1)),
                "new_in_2": zeros((1, 1)),
                "new_in_3": zeros((1, 1)),
                "new_in_4": zeros((1, 1)),
            },
        },
    )


def test_linearize_partially() -> None:
    """Check the linearization of part of the inputs/outputs of the discipline."""
    new_discipline = NewDiscipline()
    new_discipline.add_differentiated_inputs(["in_2"])
    new_discipline.add_differentiated_outputs(["out_2"])
    discipline = RemappingDiscipline(new_discipline, input_mapping, output_mapping)
    assert discipline.linearization_mode == new_discipline.linearization_mode
    discipline.linearize()
    assert_equal(
        discipline.jac,
        {
            "new_out_2": {
                "new_in_2": array([[1.0], [1.0]]),
                "new_in_3": array([[1.0], [1.0]]),
            },
        },
    )


@pytest.fixture(scope="module")
def grammar() -> SimpleGrammar:
    """A simple grammar."""
    return SimpleGrammar("X", {"x": None})


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        ({"new_in_1": "x"}, {"new_in_1": ("x", slice(None))}),
        ({"new_in_1": ("x", 1)}, {"new_in_1": ("x", slice(1, 2))}),
        ({"new_in_1": ("x", [0, 2])}, {"new_in_1": ("x", [0, 2])}),
        ({"new_in_1": ("x", range(2))}, {"new_in_1": ("x", range(2))}),
    ],
)
def test_format_mapping(mapping, expected, grammar) -> None:
    """Check the formatting of a mapping."""
    formatted_mapping = RemappingDiscipline._RemappingDiscipline__format_mapping(
        mapping, grammar
    )
    assert formatted_mapping == expected


def test_input_grammar(discipline):
    """Check the input grammar of the remapping discipline."""
    assert discipline.io.input_grammar._validate(
        {
            "new_in_1": array([1.0]),
            "new_in_2": array([2.0]),
            "new_in_3": array([3.0]),
            "new_in_4": array(["zero"]),
        },
        "",
    )


def test_output_grammar(discipline):
    """Check the output grammar of the remapping discipline."""
    assert discipline.io.output_grammar._validate(
        {
            "new_out_1": array([2.0]),
            "new_out_2": array([1.0, 2.0]),
            "new_out_3": array(["zero plus one"]),
        },
        "",
    )


def test_no_mapping():
    """Check the remapping discipline without remapping."""
    original_discipline = NewDiscipline()
    discipline = RemappingDiscipline(original_discipline)
    assert (
        discipline.io.input_grammar.keys()
        == original_discipline.io.input_grammar.keys()
    )
    assert (
        discipline.io.output_grammar.keys()
        == original_discipline.io.output_grammar.keys()
    )
    assert discipline._input_mapping == {
        f"in_{i}": (f"in_{i}", slice(None)) for i in range(1, 4)
    }
    assert discipline._output_mapping == {
        f"out_{i}": (f"out_{i}", slice(None)) for i in range(1, 4)
    }
