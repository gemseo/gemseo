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

import pytest
from numpy import array
from numpy import ones
from numpy import zeros
from scipy.optimize import rosen

from gemseo.core.mdofunctions.func_operations import LinearComposition
from gemseo.core.mdofunctions.func_operations import RestrictedFunction
from gemseo.core.mdofunctions.mdo_discipline_adapter_generator import (
    MDODisciplineAdapterGenerator,
)
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.optimization.rosen_mf import RosenMF


@pytest.mark.parametrize(
    ("input_names", "expected_expr"),
    [(["x"], "foo(A.x)"), (["x1", "x2"], "foo(A.(x1, x2)')")],
)
def test_linear_composition_expr(input_names, expected_expr):
    """Check the expression of a LinearCombination."""
    linear_composition = LinearComposition(
        MDOFunction(lambda x: x, "foo", input_names=input_names), array([[1]])
    )
    assert linear_composition.expr == expected_expr


def test_linear_composition():
    fg = MDODisciplineAdapterGenerator(RosenMF(3))
    f1 = fg.get_function(["x"], ["rosen"], default_inputs={"fidelity": 0})
    f2 = fg.get_function(["x"], ["rosen"], default_inputs={"fidelity": 1})

    x = zeros(3)
    assert f1(x) == 0.0
    assert f2(x) == rosen(x)

    interp_op = array([[0.3], [0.4], [0.5]])
    f_1_1 = LinearComposition(f1, interp_op)
    f_1_2 = LinearComposition(f2, interp_op)
    f_1_1.check_grad(ones(1), error_max=1e-4)
    f_1_2.check_grad(ones(1), error_max=1e-4)


def test_restricted_function():
    fg = MDODisciplineAdapterGenerator(RosenMF(3))
    x = zeros(3)
    f_ref = fg.get_function(["fidelity", "x"], ["rosen"])

    f1 = RestrictedFunction(
        f_ref, restriction_indices=array([0]), restriction_values=array([0])
    )

    f2 = RestrictedFunction(
        f_ref, restriction_indices=array([0]), restriction_values=array([1])
    )

    assert f1(x) == 0.0
    assert f2(x) == 2.0

    f1.check_grad(x, error_max=1e-4)
    f2.check_grad(x, error_max=1e-4)

    with pytest.raises(
        ValueError, match="Inconsistent shapes for restriction values and indices."
    ):
        RestrictedFunction(
            f_ref, restriction_indices=array([0, 1]), restriction_values=array([0])
        )
