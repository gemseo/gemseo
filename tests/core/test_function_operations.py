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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re

import pytest
from numpy import array
from numpy import ones
from numpy import zeros
from scipy.optimize import rosen

from gemseo.core.mdo_functions.discipline_adapter_generator import (
    DisciplineAdapterGenerator,
)
from gemseo.core.mdo_functions.linear_composite_function import LinearCompositeFunction
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.restricted_function import RestrictedFunction
from gemseo.problems.optimization.rosen_mf import RosenMF


@pytest.mark.parametrize(
    ("input_names", "expected_expr"),
    [(["x"], "foo(A.x)"), (["x1", "x2"], "foo(A.(x1, x2)')")],
)
def test_linear_composition_expr(input_names, expected_expr):
    """Check the expression of a LinearCombination."""
    linear_composition = LinearCompositeFunction(
        MDOFunction(lambda x: x, "foo", input_names=input_names), array([[1]])
    )
    assert linear_composition.name == "[foo o A]"
    assert linear_composition.expr == expected_expr


def test_linear_composition():
    fg = DisciplineAdapterGenerator(RosenMF(3))
    f1 = fg.get_function(["x"], ["rosen"], default_input_data={"fidelity": 0})
    f2 = fg.get_function(["x"], ["rosen"], default_input_data={"fidelity": 1})

    x = zeros(3)
    assert f1.evaluate(x) == 0.0
    assert f2.evaluate(x) == rosen(x)

    interp_op = array([[0.3], [0.4], [0.5]])
    f_1_1 = LinearCompositeFunction(f1, interp_op)
    f_1_2 = LinearCompositeFunction(f2, interp_op)
    f_1_1.check_grad(ones(1), error_max=1e-4)
    f_1_2.check_grad(ones(1), error_max=1e-4)


def test_restricted_function():
    fg = DisciplineAdapterGenerator(RosenMF(3))
    x = zeros(3)
    f_ref = fg.get_function(["fidelity", "x"], ["rosen"])

    f1 = RestrictedFunction(
        f_ref, restriction_indices=array([0]), restriction_values=array([0])
    )

    f2 = RestrictedFunction(
        f_ref, restriction_indices=array([0]), restriction_values=array([1])
    )

    assert f1.evaluate(x) == 0.0
    assert f2.evaluate(x) == 2.0

    f1.check_grad(x, error_max=1e-4)
    f2.check_grad(x, error_max=1e-4)

    with pytest.raises(
        ValueError,
        match=re.escape("Inconsistent shapes for restriction values and indices."),
    ):
        RestrictedFunction(
            f_ref, restriction_indices=array([0, 1]), restriction_values=array([0])
        )
