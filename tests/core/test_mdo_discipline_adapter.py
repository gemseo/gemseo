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

import re
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import ndarray
from numpy.testing import assert_equal

from gemseo.core.mdo_functions.discipline_adapter import DisciplineAdapter
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.core.grammars.grammar_properties import GrammarProperties
    from gemseo.core.mdo_functions.mdo_function import MDOFunction

INPUT_VECTOR = array([1.0, 1.0])


def create_disciplinary_function(
    default_input_data: GrammarProperties = READ_ONLY_EMPTY_DICT,
    names_to_sizes: Mapping[str, int] = READ_ONLY_EMPTY_DICT,
) -> DisciplineAdapter:
    """Create a disciplinary function.

    Args:
        default_input_data: The default inputs passed at instantiation.
        names_to_sizes: The input sizes passed at instantiation.
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
        names_to_sizes=names_to_sizes,
    )


@pytest.fixture
def disciplinary_function() -> DisciplineAdapter:
    """A disciplinary function."""
    return create_disciplinary_function()


def check_func_and_jac_evaluation(mdo_function: MDOFunction) -> None:
    """Check the evaluation of the function and its Jacobian."""
    assert_equal(mdo_function.evaluate(INPUT_VECTOR), array([2.0]))
    assert_equal(mdo_function.jac(INPUT_VECTOR), array([1.0, 1.0]))


def test_error(disciplinary_function) -> None:
    """Check that a ValueError is raised when the size of an input cannot be guessed."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The size of the input x cannot be guessed from the discipline my_func, "
            "nor from its default inputs or from its local data."
        ),
    ):
        disciplinary_function.func(INPUT_VECTOR)


def test_discipline_local_data(disciplinary_function) -> None:
    """Check that input sizes can be guessed from the discipline's local data."""
    disciplinary_function._DisciplineAdapter__discipline.io.data.update({
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
    disciplinary_function = create_disciplinary_function(names_to_sizes={"x": 1})
    check_func_and_jac_evaluation(disciplinary_function)
