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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import ndarray
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo.core.discipline import Discipline
from gemseo.core.function.discipline_adapter import DisciplineAdapter
from gemseo.discipline.auto_py import AutoPyDiscipline
from gemseo.util.constant import read_only_empty_dict
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.core.grammar.properties import GrammarProperties

input_vector = array([1.0, 1.0])


def create_disciplinary_function(
    default_input_data: GrammarProperties = read_only_empty_dict,
    name_to_size: Mapping[str, int] = read_only_empty_dict,
) -> DisciplineAdapter:
    """Create a disciplinary function.

    Args:
        default_input_data: The default inputs passed at instantiation.
        name_to_size: The input sizes passed at instantiation.
    """

    def my_func(x: float, y: float = 0.0) -> float:
        z = x + y
        return z  # noqa: RET504

    def my_jac(x: float, y: float = 0.0) -> ndarray:
        return array([[1.0, 1.0]])

    discipline = AutoPyDiscipline(my_func, py_jac=my_jac)
    discipline.add_differentiated_inputs(["x", "y"])
    discipline.add_differentiated_outputs(["z"])
    return DisciplineAdapter(
        ["x", "y"],
        ["z"],
        default_input_data or {},  # Because READ_ONLY_EMPTY_DICT cannot be pickled.
        discipline,
        name_to_size=name_to_size,
    )


@pytest.fixture
def disciplinary_function() -> DisciplineAdapter:
    """A disciplinary function."""
    return create_disciplinary_function()


def check_func_and_jac_evaluation(mdo_function: ArrayFunction) -> None:
    """Check the evaluation of the function and its Jacobian."""
    assert_equal(mdo_function.evaluate(input_vector), array([2.0]))
    assert_equal(mdo_function.jac(input_vector), array([1.0, 1.0]))


def test_error(disciplinary_function, snapshot) -> None:
    """Check that a ValueError is raised when the size of an input cannot be guessed."""
    with assert_exception(ValueError, snapshot):
        disciplinary_function.func(input_vector)


def test_error_several_inputs(snapshot) -> None:
    """Check the error message when several input sizes cannot be guessed."""

    def my_func(x: float, y: float) -> float:
        z = x + y
        return z  # noqa: RET504

    function = DisciplineAdapter(["x", "y"], ["z"], {}, AutoPyDiscipline(my_func))
    with assert_exception(ValueError, snapshot):
        function.func(input_vector)


def test_discipline_local_data(disciplinary_function) -> None:
    """Check that input sizes can be guessed from the discipline's local data."""
    disciplinary_function._DisciplineAdapter__discipline.io.input_data.update({
        "x": array([1.0])
    })
    check_func_and_jac_evaluation(disciplinary_function)


def test_discipline_default_inputs(disciplinary_function) -> None:
    """Check that input sizes can be guessed from the discipline's default inputs."""
    disciplinary_function._DisciplineAdapter__discipline.io.input_grammar.defaults.update({
        "x": array([1.0])
    })
    check_func_and_jac_evaluation(disciplinary_function)


def test_default_inputs() -> None:
    """Check that input sizes can be guessed from the function's default inputs."""
    disciplinary_function = create_disciplinary_function(
        default_input_data={"x": array([1.0])}
    )
    check_func_and_jac_evaluation(disciplinary_function)


def test_names_to_sizes() -> None:
    """Check that input sizes can be guessed from the function's input sizes."""
    disciplinary_function = create_disciplinary_function(name_to_size={"x": 1})
    check_func_and_jac_evaluation(disciplinary_function)


def test_error_differentiated_input(snapshot) -> None:
    """Check the error message when the size of a differentiated input is unknown."""

    def my_func(y: float, x: float = 0.0) -> float:
        z = x + y
        return z  # noqa: RET504

    function = DisciplineAdapter(
        ["x"],
        ["z"],
        {},
        AutoPyDiscipline(my_func),
        differentiated_input_names_substitute=["y"],
    )
    with assert_exception(ValueError, snapshot):
        function.func(array([1.0]))


class DisciplineWithNonNumericInput(Discipline):
    """A discipline with a non-numeric input that is not a function input."""

    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_types({"x": list, "z": float})
        self.output_grammar.update_from_types({"f": float})
        self.io.input_grammar.defaults.update({"x": [1.0], "z": 0.5})

    def _run(self, input_data):
        x = input_data["x"][0]
        z = input_data["z"]
        return {"f": x * z}


def test_non_numeric_input_not_in_input_names() -> None:
    """Check that a non-numeric, non-differentiated input does not break the adapter.

    A discipline input whose value is not a number or an array (e.g. a `list`)
    used to make the adapter crash while computing sizes for all the discipline
    inputs, even though the adapter only needs the sizes of `input_names` and
    `differentiated_input_names_substitute`.
    """
    discipline = DisciplineWithNonNumericInput()
    function = DisciplineAdapter(["z"], ["f"], {}, discipline)
    assert_equal(function.evaluate(array([0.5])), array([0.5]))


class _DisciplineWithSummingInput(Discipline):
    """A discipline whose single input `x` defaults to a size-1 array."""

    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_types({"x": ndarray})
        self.output_grammar.update_from_types({"f": float})
        self.io.input_grammar.defaults.update({"x": array([1.0])})

    def _run(self, input_data):
        return {"f": float(input_data["x"].sum())}


def test_shared_name_to_size_prevails_over_discipline_size() -> None:
    """Check that a size shared with the formulation prevails over the discipline's.

    The `name_to_size` mapping passed at instantiation may be shared with the
    formulation, which seeds it with the design variable sizes. This size must
    be used to slice the input vector, even though it differs from the size
    that would be measured from the discipline's default input value.
    """
    discipline = _DisciplineWithSummingInput()
    function = DisciplineAdapter(["x"], ["f"], {}, discipline, name_to_size={"x": 3})
    assert_equal(function.evaluate(array([1.0, 2.0, 3.0])), array([6.0]))


class _DisciplineWithSquaringInput(Discipline):
    """A discipline summing the squares of its input `x`, defaulting to a size-1 array.

    Its Jacobian is approximated by finite differences.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_types({"x": ndarray})
        self.output_grammar.update_from_types({"f": float})
        self.io.input_grammar.defaults.update({"x": array([1.0])})
        self.linearization_mode = "finite_differences"

    def _run(self, input_data):
        return {"f": float((input_data["x"] ** 2).sum())}


def test_local_data_size_prevails_over_default_size() -> None:
    """Check that an input size is measured from the local data before the defaults.

    The default input values of the discipline used to prevail over its local data
    when measuring the input sizes. An approximated Jacobian was then silently
    truncated to the size of the default value: a discipline defaulting to an `x`
    of size 1 but linearized at an `x` of size 3 returned a Jacobian of shape
    `(1, 1)` instead of `(1, 3)`. The local data is more recent than the default
    input values, hence it must prevail over them.
    """
    discipline = _DisciplineWithSquaringInput()
    discipline.add_differentiated_inputs(["x"])
    discipline.add_differentiated_outputs(["f"])
    jacobian = discipline.linearize({"x": array([1.0, 2.0, 3.0])})["f"]["x"]
    assert jacobian.shape == (1, 3)
    assert_allclose(jacobian, array([[2.0, 4.0, 6.0]]), atol=1e-6)


class _DisciplineWithTwoInputs(Discipline):
    """A discipline whose inputs `x` and `y` default to size-1 arrays.

    The output `f` weights `x` and `y` differently so that a wrong split of the
    input vector between the two gives a different value rather than the same one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_types({"x": ndarray, "y": ndarray})
        self.output_grammar.update_from_types({"f": float})
        self.io.input_grammar.defaults.update({"x": array([1.0]), "y": array([1.0])})

    def _run(self, input_data):
        return {"f": float(input_data["x"].sum() + 10.0 * input_data["y"].sum())}


def test_default_inputs_size_prevails_over_local_data_size() -> None:
    """Check that the adapter's default input data prevails over the local data.

    The discipline's local data may be stale, e.g. left over from a previous
    evaluation at another point, and of another size than the point at which the
    adapter is now evaluated. The adapter's own `default_input_data`, passed at
    instantiation, is the most trustworthy source of sizes and must prevail over
    the discipline's local data when the two disagree on the size of an input.
    """
    discipline = _DisciplineWithTwoInputs()
    discipline.io.input_data.update({"x": array([9.0])})
    function = DisciplineAdapter(
        ["x", "y"], ["f"], {"x": array([1.0, 2.0])}, discipline
    )
    assert_equal(function.evaluate(array([1.0, 2.0, 3.0])), array([33.0]))
