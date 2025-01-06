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
# INITIAL AUTHORS - initial API and implementation and/or
#                   initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for analytic Discipline based on symbolic expressions."""

from __future__ import annotations

import re

import pytest
import sympy
from numpy import array
from numpy.testing import assert_equal

from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.pickle import from_pickle
from gemseo.utils.pickle import to_pickle


@pytest.fixture
def expressions():
    # string expressions
    expr_dict = {"y_1": "2*x**2", "y_2": "3*x**2+5+z**3"}
    # SymPy expression
    x, z = sympy.symbols(["x", "z"])
    y_3 = sympy.Piecewise(
        (sympy.exp(-1 / (1 - x**2 - z**2)), x**2 + z**2 < 1), (0, True)
    )
    expr_dict["y_3"] = y_3
    # N.B. y_3 is infinitely differentiable with respect to x and z
    return expr_dict


def test_independent_default_inputs() -> None:
    """Test that the default inputs are independent.

    Reproducer for #406.
    """
    expr = {"obj": "x1 + x2 + x3"}
    disc = AnalyticDiscipline(expr)
    disc.execute()
    disc.io.data["x1"] += 1.0
    assert disc.io.data["x2"] == pytest.approx(0.0)


def test_fast_expression_evaluation(expressions) -> None:
    disc = AnalyticDiscipline(expressions)
    input_data = {"x": array([1.0]), "z": array([1.0])}
    disc.check_jacobian(input_data, step=1e-5, threshold=1e-3)


def test_failure_with_malformed_expressions() -> None:
    with pytest.raises(
        TypeError, match=re.escape("Expression must be a SymPy expression or a string.")
    ):
        AnalyticDiscipline({"y": MDOScenario})


def test_absolute_value() -> None:
    """Check that AnalyticDiscipline handles absolute value."""
    discipline = AnalyticDiscipline({"y": "Abs(x)"})
    assert (
        discipline.linearize({"x": array([2])}, compute_all_jacobians=True)["y"]["x"]
        == 1
    )
    assert (
        discipline.linearize({"x": array([-2])}, compute_all_jacobians=True)["y"]["x"]
        == -1
    )

    # Be careful: the derivative of sympy.Abs(x) at 0 is equal to 0
    # even if it is not defined at 0 from a mathematical point of view.
    assert (
        discipline.linearize({"x": array([0])}, compute_all_jacobians=True)["y"]["x"]
        == 0
    )


def test_serialize(tmp_wd) -> None:
    """Check the serialization of an AnalyticDiscipline."""
    input_data = {"x": array([2.0])}
    file_path = "discipline.h5"

    discipline = AnalyticDiscipline({"y": "2*x"})
    to_pickle(discipline, file_path)
    discipline.execute(input_data)

    saved_discipline = from_pickle(file_path)
    saved_discipline.execute(input_data)

    assert_equal(saved_discipline.io.get_output_data(), discipline.io.get_output_data())


@pytest.mark.parametrize("add_differentiated_inputs", [False, True])
@pytest.mark.parametrize("add_differentiated_outputs", [False, True])
@pytest.mark.parametrize("compute_all_jacobians", [False, True])
def test_linearize(
    add_differentiated_inputs,
    add_differentiated_outputs,
    compute_all_jacobians,
    caplog,
) -> None:
    """Check AnalyticDiscipline.linearize()."""
    discipline = AnalyticDiscipline({"y": "2*a+3*b", "z": "-2*a-3*b"})
    if add_differentiated_inputs:
        discipline.add_differentiated_inputs(["a"])
    if add_differentiated_outputs:
        discipline.add_differentiated_outputs(["y"])

    discipline.linearize(
        input_data={"a": array([1]), "b": array([1])},
        compute_all_jacobians=compute_all_jacobians,
    )
    if compute_all_jacobians:
        assert discipline.jac == {
            "y": {"a": array([[2.0]]), "b": array([[3.0]])},
            "z": {"a": array([[-2.0]]), "b": array([[-3.0]])},
        }
    elif add_differentiated_inputs and add_differentiated_outputs:
        assert discipline.jac == {"y": {"a": array([[2.0]])}}
    else:
        assert discipline.jac == {}


def test_complex_outputs() -> None:
    """Check that complex outputs are supported."""
    discipline = AnalyticDiscipline({"y": "x*I"})
    discipline.execute({"x": array([1.0])})
    assert discipline.io.data["y"] == 1j


@pytest.mark.parametrize("linearization_mode", ApproximationMode)
def test_jacobian_approximation(linearization_mode) -> None:
    """Check that Jacobian approximation is supported."""
    discipline = AnalyticDiscipline({"y": "exp(x)"})
    discipline.linearization_mode = linearization_mode
    discipline.linearize({"x": array([0.0])}, True)
    assert discipline.jac["y"]["x"] == pytest.approx(1)
