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
# Copyright 2022 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.api import create_discipline
from gemseo.api import create_mda
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.core.discipline import MDODiscipline
from numpy import array
from numpy import ndarray
from pytest import fixture


def disc_1_expr(w1: float = 0.0, y2: float = 2.0, x: float = 3.0) -> tuple[float]:
    """A linear function with residuals. First toy discipline.

    Args:
        w1: The initial state value.
        y2: The input coupling value.
        x: The design variables.

    Returns:
        The couplings1, state1 and residuals1
    """
    w1 = (3 * x - y2) / 7.0
    y1 = 5 * w1 + x + 3 * y2
    r1 = y2 - 3 * x + 7 * w1
    return y1, w1, r1


def disc_1_expr_jac(w1: float = 0.0, y2: float = 2.0, x: float = 3.0) -> ndarray:
    """The Jacobian of linear function with residuals. First toy discipline.

    Args:
        w1: The initial state value.
        y2: The input coupling value.
        x: The design variables.

    Returns:
        The couplings1, state1 and residuals1 Jacobian.
    """
    d_y_w_r_d_w_y_x = array(
        [
            [5.0, 3.0, 1.0],
            [0.0, -1.0 / 7, 3.0 / 7],
            [7, 1, -3],
        ]
    )
    return d_y_w_r_d_w_y_x


def disc_2_expr(w2: float = 3.0, y1: float = 1.0, x: float = 2.0) -> tuple[float]:
    """A linear function with residuals. Second toy discipline.

    Args:
        w2: The initial state value.
        y1: The input coupling value.
        x: The design variables.

    Returns:
        The couplings2, state2 and radius2
    """
    w2 = (2 * x - y1) / 5.0
    y2 = 13 * w2 + x + 2 * y1

    r2 = y1 - 2 * x + 5 * w2
    return y2, w2, r2


def disc_2_expr_jac(w2: float = 3.0, y1: float = 1.0, x: float = 2.0) -> ndarray:
    """The Jacobian of linear function with residuals. Second toy discipline.

    Args:
        w2: The initial state value.
        y1: The input coupling value.
        x: The design variables.

    Returns:
        The couplings2, state2 and residuals2 Jacobian.
    """
    d_y_w_r_d_w_y_x = array(
        [
            [13.0, 2.0, 1.0],
            [0.0, -1.0 / 5.0, 2 / 5],
            [5, 1, -2],
        ]
    )
    return d_y_w_r_d_w_y_x


def disc_3_expr(y1: float = 1.0, y2: float = 2.0, x: float = 2.0) -> float:
    """A linear objective function. Third toy discipline.

    Args:
        y1: The first input coupling value.
        y2: The second input coupling value.
        x: The design variables.

    Returns:
        The objective function.
    """
    obj = y1 + 2 * y2 + 11 * x
    return obj


def disc_3_expr_jac(y1: float = 1.0, y2: float = 2.0, x: float = 2.0) -> ndarray:
    """A linear objective function. Third toy discipline.

    Args:
        y1: The first input coupling value.
        y2: The second input coupling value.
        x: The design variables.

    Returns:
        The objective function Jacobian.
    """
    d_obj_d_y1_y2_x = array([[1.0, 2.0, 11.0]])
    return d_obj_d_y1_y2_x


@fixture
def res_disciplines() -> list[MDODiscipline]:
    """Create the three disciplines required to make a MDA with residual variables.

    Returns:
        The disciplines instances.
    """
    d1 = create_discipline(
        "AutoPyDiscipline", py_func=disc_1_expr, py_jac=disc_1_expr_jac
    )
    d1.residual_variables = {"r1": "w1"}
    d1.run_solves_residuals = True
    d2 = create_discipline(
        "AutoPyDiscipline", py_func=disc_2_expr, py_jac=disc_2_expr_jac
    )
    d2.residual_variables = {"r2": "w2"}
    d2.run_solves_residuals = True
    d3 = create_discipline(
        "AutoPyDiscipline", py_func=disc_3_expr, py_jac=disc_3_expr_jac
    )

    return [d1, d2, d3]


def test_residuals_mda(res_disciplines):
    """Test MDA execution with residuals variables in disciplines."""
    coupling_structure = MDOCouplingStructure(res_disciplines)
    for disc in res_disciplines:
        assert not coupling_structure.is_self_coupled(disc)
    mda = create_mda("MDAChain", disciplines=res_disciplines)
    out = mda.execute()
    assert out["r1"] < 1e-13
    assert out["r2"] < 1e-13

    for disc in res_disciplines:
        disc.linearize(force_all=True)

    assembly = JacobianAssembly(MDOCouplingStructure(res_disciplines))
    assembly.compute_sizes(
        ["obj"],
        variables=["x"],
        couplings=["y1", "y2", "w1", "w2"],
        residual_variables={"r1": "w1", "r2": "w2"},
    )
    for var in ["y1", "y2", "w1", "w2", "obj", "x", "r1", "r2"]:
        assert assembly.sizes[var] == 1


@pytest.mark.parametrize(
    "mode",
    [JacobianAssembly.ADJOINT_MODE, JacobianAssembly.DIRECT_MODE],
)
@pytest.mark.parametrize(
    "matrix_type",
    [JacobianAssembly.SPARSE, JacobianAssembly.LINEAR_OPERATOR],
)
def test_adjoint(res_disciplines, mode, matrix_type):
    """Test the coupled adjoint with residual variables in disciplines."""
    mda = create_mda("MDAChain", disciplines=res_disciplines)
    mda.linearization_mode = mode
    mda.matrix_type = matrix_type
    assert mda.check_jacobian(inputs=["x"], outputs=["obj"])
