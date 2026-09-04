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

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import zeros
from numpy.testing import assert_equal

from gemseo.core.discipline import Discipline
from gemseo.core.grammar.simple import SimpleGrammar
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.discipline.auto_py import AutoPyDiscipline
from gemseo.discipline.remapping import RemappingDiscipline
from gemseo.util.discipline import DummyDiscipline
from gemseo.util.pickle import from_pickle
from gemseo.util.pickle import to_pickle
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.util.typing import StrKeyMapping


class NoInputsDiscipline(Discipline):
    """A discipline without input variables."""

    def __init__(self):
        super().__init__()
        self.output_grammar.update_from_names(["foo"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {"foo": array([10.0])}


class NoOutputsDiscipline(Discipline):
    """A discipline without output variables."""

    def __init__(self):
        super().__init__()
        self.input_grammar.update_from_names(["foo"])
        self.io.input_grammar.defaults["foo"] = array([0.0])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {}


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
        self.io.output_data["out_1"] = input_data["in_1"] + 1
        self.io.output_data["out_2"] = input_data["in_2"] - 1
        self.io.output_data["out_3"] = array([f"{input_data['in_3'][0]} plus one"])

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
    """Check that the wrapped discipline needs no default input value."""
    discipline = DummyDiscipline(input_names=["x"])
    remapping_discipline = RemappingDiscipline(discipline, {"new_x": "x"})
    assert remapping_discipline.io.input_grammar.keys() == {"new_x"}
    assert not remapping_discipline.io.input_grammar.defaults


@pytest.mark.parametrize("use_default", [False, True])
def test_component_mapping_wo_default_values(use_default, snapshot) -> None:
    """Check that an input mapped component-wise must have a default value."""
    discipline = DummyDiscipline(input_names=["x", "y"])
    if use_default:
        discipline.io.input_grammar.defaults["x"] = array([0.0, 1.0])
    with assert_exception(ValueError, snapshot):
        RemappingDiscipline(discipline, {"new_x": ("x", 0), "new_y": ("y", 0)}, {})


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
    assert_equal(discipline.io.output_data["new_out_1"], array([2.0]))
    assert_equal(discipline.io.output_data["new_out_2"], array([1.0, 2.0]))


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


def test_no_inputs_discipline():
    """Check that RemappingDiscipline supports disciplines without input variables."""
    discipline = NoInputsDiscipline()
    remapping_discipline = RemappingDiscipline(
        discipline=discipline,
        output_mapping={"bar": "foo"},
    )
    remapping_discipline.execute()
    assert_equal(remapping_discipline.io.output_data["bar"], array([10.0]))


def test_no_outputs_discipline():
    """Check that RemappingDiscipline supports disciplines without output variables."""
    discipline = NoOutputsDiscipline()
    remapping_discipline = RemappingDiscipline(
        discipline=discipline, input_mapping={"bar": "foo"}
    )
    remapping_discipline.execute()
    assert not remapping_discipline.get_output_data()
    assert_equal(discipline.io.input_data["foo"], array([0.0]))
    remapping_discipline.execute({"bar": array([10.0])})
    assert not remapping_discipline.get_output_data()
    assert_equal(discipline.io.input_data["foo"], array([10.0]))


def test_linearize():
    """Check if the linearize method is correctly handled in a RemappingDiscipline."""
    old_discipline = AnalyticDiscipline({"old_a": "0.5 * old_b + old_c"})
    old_jac = old_discipline.linearize(
        {"old_b": array([2.0]), "old_c": array([1.0])}, compute_all_jacobians=True
    )
    new_discipline = RemappingDiscipline(
        old_discipline,
        input_mapping={"new_b": "old_b", "new_c": "old_c"},
        output_mapping={"new_a": "old_a"},
    )
    new_jac = new_discipline.linearize(
        {"new_b": array([2.0]), "new_c": array([1.0])}, compute_all_jacobians=True
    )
    assert_equal(old_jac["old_a"]["old_b"], new_jac["new_a"]["new_b"])
    assert_equal(old_jac["old_a"]["old_c"], new_jac["new_a"]["new_c"])


def add_arrays(a: ndarray, b: ndarray) -> ndarray:
    """Sum two variables without default values.

    Args:
        a: The first operand.
        b: The second operand.

    Returns:
        The sum of the operands.
    """
    c = a + b
    return c  # noqa: RET504


def add_floats(a: float = 2.0, b: float = 3.0) -> float:
    """Sum two float variables with default values.

    Args:
        a: The first operand.
        b: The second operand.

    Returns:
        The sum of the operands.
    """
    c = a + b
    return c  # noqa: RET504


def test_auto_py_discipline_wo_default_values():
    """Check the remapping of an AutoPyDiscipline without default input values."""
    discipline = RemappingDiscipline(
        AutoPyDiscipline(add_arrays),
        input_mapping={"x": "a", "y": "b"},
    )
    assert not discipline.io.input_grammar.defaults
    discipline.execute({"x": array([1.0]), "y": array([2.0])})
    assert_equal(discipline.io.output_data["c"], array([3.0]))


def test_non_array_values():
    """Check the remapping of a discipline handling non-array values."""
    discipline = RemappingDiscipline(
        AutoPyDiscipline(add_floats),
        input_mapping={"x": "a", "y": "b"},
    )
    assert discipline.io.input_grammar.defaults == {"x": 2.0, "y": 3.0}
    discipline.execute()
    assert discipline.io.output_data["c"] == 5.0
    discipline.execute({"x": 1.0})
    assert discipline.io.output_data["c"] == 4.0


def test_unmapped_input_keeps_its_default_value():
    """Check that an input missing from the mapping keeps its default value."""
    original_discipline = AnalyticDiscipline({"z": "a + b"})
    original_discipline.io.input_grammar.defaults.update({
        "a": array([1.0]),
        "b": array([100.0]),
    })
    discipline = RemappingDiscipline(original_discipline, input_mapping={"x": "a"})
    discipline.execute({"x": array([1.0])})
    assert_equal(original_discipline.io.input_data["b"], array([100.0]))
    assert_equal(discipline.io.output_data["z"], array([101.0]))
