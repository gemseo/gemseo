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
"""Tests for analytic MDODiscipline based on symbolic expressions."""
from __future__ import annotations

import pytest
import sympy
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from numpy import array
from numpy.testing import assert_equal
from packaging import version


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


def test_independent_default_inputs():
    """Test that the default inputs are independent.

    Reproducer for #406.
    """
    expr = {"obj": "x1 + x2 + x3"}
    disc = AnalyticDiscipline(expr)
    disc.execute()
    disc.local_data["x1"] += 1.0
    assert disc.local_data["x2"] == pytest.approx(0.0)


def test_fast_expression_evaluation(expressions):
    disc = AnalyticDiscipline(expressions)
    input_data = {"x": array([1.0]), "z": array([1.0])}
    disc.check_jacobian(input_data, step=1e-5, threshold=1e-3)


def test_standard_expression_evaluation(expressions):
    disc = AnalyticDiscipline(expressions, fast_evaluation=False)
    input_data = {"x": array([1.0]), "z": array([1.0])}
    disc.check_jacobian(input_data, step=1e-5, threshold=1e-3)


def test_failure_with_malformed_expressions():
    with pytest.raises(
        TypeError, match="Expression must be a SymPy expression or a string."
    ):
        AnalyticDiscipline({"y": MDOScenario})


@pytest.mark.skipif(
    version.parse(sympy.__version__) > version.parse("1.8.0"),
    reason="requires sympy 1.7.0 or lower",
)
def test_failure_for_log_zero_without_fast_evaluation():
    # For sympy 1.8.0 and higher,
    # sympy.parsing.sympy_parser.parse_expr("log(x)").evalf(subs={"x":0.0})
    # returns -oo which is converted into the float -inf."""
    disc = AnalyticDiscipline({"y": "log(x)"}, fast_evaluation=False)
    input_data = {"x": array([0.0])}
    with pytest.raises(TypeError):
        disc.execute(input_data)


@pytest.mark.parametrize("fast_evaluation", [False, True])
def test_absolute_value(fast_evaluation):
    """Check that AnalyticDiscipline handles absolute value."""
    discipline = AnalyticDiscipline({"y": "Abs(x)"}, fast_evaluation=fast_evaluation)
    assert discipline.linearize({"x": array([2])}, force_all=True)["y"]["x"] == 1
    assert discipline.linearize({"x": array([-2])}, force_all=True)["y"]["x"] == -1

    # Be careful: the derivative of sympy.Abs(x) at 0 is equal to 0
    # even if it is not defined at 0 from a mathematical point of view.
    assert discipline.linearize({"x": array([0])}, force_all=True)["y"]["x"] == 0


@pytest.mark.parametrize("fast_evaluation", [False, True])
def test_serialize(tmp_wd, fast_evaluation):
    """Check the serialization of an AnalyticDiscipline."""
    input_data = {"x": array([2.0])}
    file_path = "discipline.h5"

    discipline = AnalyticDiscipline({"y": "2*x"}, fast_evaluation=fast_evaluation)
    discipline.serialize(file_path)
    discipline.execute(input_data)

    saved_discipline = AnalyticDiscipline.deserialize(file_path)
    saved_discipline.execute(input_data)

    assert_equal(saved_discipline.get_output_data(), discipline.get_output_data())
