# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

import pytest
import sympy
from numpy import array
from packaging import version

from gemseo.core.mdo_scenario import MDOScenario
from gemseo.disciplines.analytic import AnalyticDiscipline


@pytest.fixture
def expressions():
    # string expressions
    expr_dict = {"y_1": "2*x**2", "y_2": "3*x**2+5+z**3"}
    # SymPy expression
    x, z = sympy.symbols(["x", "z"])
    y_3 = sympy.Piecewise(
        (sympy.exp(-1 / (1 - x ** 2 - z ** 2)), x ** 2 + z ** 2 < 1), (0, True)
    )
    expr_dict["y_3"] = y_3
    # N.B. y_3 is infinitely differentiable with respect to x and z
    return expr_dict


def test_fast_expression_evaluation(expressions):
    disc = AnalyticDiscipline(expressions)
    input_data = {"x": array([1.0]), "z": array([1.0])}
    disc.check_jacobian(
        input_data, derr_approx=disc.FINITE_DIFFERENCES, step=1e-5, threshold=1e-3
    )


def test_standard_expression_evaluation(expressions):
    disc = AnalyticDiscipline(expressions, fast_evaluation=False)
    input_data = {"x": array([1.0]), "z": array([1.0])}
    disc.check_jacobian(
        input_data, derr_approx=disc.FINITE_DIFFERENCES, step=1e-5, threshold=1e-3
    )


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
