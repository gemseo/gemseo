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
from typing import Mapping

import pytest
from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator
from gemseo.core.mdofunctions.make_function import MakeFunction
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.disciplines.auto_py import AutoPyDiscipline
from numpy import array
from numpy import ndarray
from numpy.testing import assert_equal

INPUT_VECTOR = array([1.0, 1.0])


def create_disciplinary_function(
    default_inputs: Mapping[str, ndarray] | None = None,
    names_to_sizes: Mapping[str, int] | None = None,
) -> MakeFunction:
    """Create a disciplinary function.

    Args:
        default_inputs: The default inputs passed at instantiation.
        names_to_sizes: The input sizes passed at instantiation.
    """

    def my_func(x: float, y: float = 0.0) -> ndarray:
        z = x + y
        return z

    def my_jac(x: float, y: float = 0.0) -> ndarray:
        return array([[1.0, 1.0]])

    discipline = AutoPyDiscipline(my_func, py_jac=my_jac)
    discipline.add_differentiated_outputs(["z"])
    return MakeFunction(
        ["x", "y"],
        ["z"],
        default_inputs=default_inputs,
        mdo_function=MDOFunctionGenerator(discipline),
        names_to_sizes=names_to_sizes,
    )


@pytest.fixture
def disciplinary_function() -> MakeFunction:
    """A disciplinary function."""
    return create_disciplinary_function()


def check_func_and_jac_evaluation(mdo_function: MDOFunction) -> None:
    """Check the evaluation of the function and its Jacobian."""
    assert_equal(mdo_function.func(INPUT_VECTOR), array([2.0]))
    assert_equal(mdo_function.jac(INPUT_VECTOR), array([1.0, 1.0]))


def test_error(disciplinary_function):
    """Check that a ValueError is raised when the size of an input cannot be guessed."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The size of the input x cannot be guessed from the discipline my_func, "
            "nor from its default inputs or from its local data."
        ),
    ):
        disciplinary_function.func(INPUT_VECTOR)


def test_discipline_local_data(disciplinary_function):
    """Check that input sizes can be guessed from the discipline's local data."""
    disciplinary_function._MakeFunction__discipline.local_data.update(
        {"x": array([1.0])}
    )
    check_func_and_jac_evaluation(disciplinary_function)


def test_discipline_default_inputs(disciplinary_function):
    """Check that input sizes can be guessed from the discipline's default inputs."""
    disciplinary_function._MakeFunction__discipline.default_inputs.update(
        {"x": array([1.0])}
    )
    check_func_and_jac_evaluation(disciplinary_function)


def test_default_inputs():
    """Check that input sizes can be guessed from the function's default inputs."""
    disciplinary_function = create_disciplinary_function(
        default_inputs={"x": array([1.0])}
    )
    check_func_and_jac_evaluation(disciplinary_function)


def test_names_to_sizes():
    """Check that input sizes can be guessed from the function's input sizes."""
    disciplinary_function = create_disciplinary_function(names_to_sizes={"x": 1})
    check_func_and_jac_evaluation(disciplinary_function)
