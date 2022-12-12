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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.derivatives import jacobian_assembly
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.process.mda_gauss_seidel import SobieskiMDAGaussSeidel
from numpy import ndarray
from numpy import random
from scipy.sparse import csr_matrix

CWD = Path(__file__).parent


@pytest.fixture(scope="module")
def assembly() -> JacobianAssembly:
    """An assembly of Jacobians."""
    disciplines = [SobieskiAerodynamics(), SobieskiMission()]
    for discipline in disciplines:
        discipline.linearize(force_all=True)

    return JacobianAssembly(MDOCouplingStructure(disciplines))


@pytest.fixture(scope="module")
def in_data() -> dict[str, ndarray]:
    """The default values of the inputs of the SSBJ problem."""
    return SobieskiProblem().get_default_inputs()


def test_total_derivatives_ok(assembly, in_data):
    """Check total_derivatives()."""
    assembly.total_derivatives(in_data, ["y_4"], ["x_shared"], ["y_24"])


@pytest.mark.parametrize(
    "args,kwargs,msg",
    [
        (
            [["toto"], ["x_shared"], ["y_24"]],
            {},
            (
                "Some outputs are not computed by the disciplines:{'toto'} "
                "available outputs are: "
                "[['y_21', 'y_23', 'y_24', 'g_2', 'y_2'], ['y_4']]"
            ),
        ),
        (
            [["y_4"], ["toto"], ["y_24"]],
            {},
            (
                "Some of the specified variables are not inputs of the disciplines: "
                "{'toto'} "
                "possible inputs are: [['x_2', 'y_32', 'x_shared', 'y_12', 'c_4'], "
                "['y_14', 'x_shared', 'y_24', 'y_34']]"
            ),
        ),
        (
            [["y_4"], ["x_shared"], ["x_shared"]],
            {"matrix_type": "foo"},
            "Variable x_shared is both a coupling and a design variable",
        ),
        (
            [["y_4"], ["x_3"], ["y_12"]],
            {"mode": "ERROR"},
            (
                "Some of the specified variables are not inputs of the disciplines: "
                "{'x_3'} "
                "possible inputs are: [['x_2', 'y_32', 'x_shared', 'y_12', 'c_4'], "
                "['y_14', 'x_shared', 'y_24', 'y_34']]"
            ),
        ),
    ],
)
def test_total_derivatives_ko(assembly, in_data, args, kwargs, msg):
    """Check that the errors raised by total_derivatives()."""
    with pytest.raises(ValueError, match=re.escape(msg)):
        assembly.total_derivatives(in_data, *args, **kwargs)


def test_compute_sizes_ko(assembly):
    """Check the consistency error raised by compute_sizes()."""
    with pytest.raises(
        ValueError,
        match=re.escape("Failed to determine the size of input variable foo"),
    ):
        assembly.compute_sizes(["y_4"], ["foo"], ["y_24"])


def compare_mda_jac_ref(jacobian: dict[str, dict[str, ndarray]]) -> bool:
    """Compare a given Jacobian with reference Jacobian in file."""
    with (Path(__file__).parent / "mda_grad_sob.json").open() as reference_jacobian:
        for ykey, jac_dict in json.load(reference_jacobian).items():
            if ykey not in jacobian:
                return False

            for xkey, jac_loc in jac_dict.items():
                if xkey not in jacobian[ykey]:
                    return False

                if not np.allclose(np.array(jac_loc), jacobian[ykey][xkey], atol=1e-1):
                    return False

    return True


@pytest.fixture(scope="module")
def functions() -> list[str]:
    """The output names."""
    return ["y_4", "g_1", "g_2", "g_3", "y_1", "y_2", "y_3"]


@pytest.fixture(scope="module")
def variables() -> list[str]:
    """The names of the design variables."""
    return ["x_shared", "x_1", "x_2", "x_3"]


@pytest.fixture(scope="module")
def couplings() -> list[str]:
    """The names of the coupling variables."""
    return ["y_23", "y_12", "y_14", "y_31", "y_24", "y_32", "y_34", "y_21"]


@pytest.fixture(scope="module")
def mda(in_data, functions, variables, couplings) -> SobieskiMDAGaussSeidel:
    """A Gauss-Seidel MDA for the SSBJ use case."""
    gs_mda = SobieskiMDAGaussSeidel("complex128")
    gs_mda.tolerance = 1e-14
    gs_mda.max_iter = 100
    gs_mda.add_differentiated_inputs(variables)
    gs_mda.add_differentiated_outputs(functions)
    gs_mda.jac = gs_mda.assembly.total_derivatives(
        in_data,
        functions,
        variables,
        couplings,
    )
    return gs_mda


@pytest.mark.parametrize(
    "mode",
    [JacobianAssembly.DIRECT_MODE, JacobianAssembly.ADJOINT_MODE],
)
@pytest.mark.parametrize(
    "matrix_type", [JacobianAssembly.SPARSE, JacobianAssembly.LINEAR_OPERATOR]
)
@pytest.mark.parametrize("use_lu_fact", [False, True])
def test_sobieski_all_modes(
    mda, in_data, functions, variables, couplings, mode, matrix_type, use_lu_fact
):
    """Test Sobieski's coupled derivatives computed in all modes (sparse direct, sparse
    adjoint, linear operator direct, linear operator adjoint)"""
    if use_lu_fact and not matrix_type == JacobianAssembly.SPARSE:
        return

    mda.jac = mda.assembly.total_derivatives(
        in_data,
        functions,
        variables,
        couplings,
        mode=mode,
        matrix_type=matrix_type,
        use_lu_fact=use_lu_fact,
    )
    if not compare_mda_jac_ref(mda.jac):
        raise Exception(
            f"Linearization mode '{mode} 'failed for matrix type "
            f"{matrix_type} and use_lu_fact ={use_lu_fact}"
        )


def test_total_derivatives(mda, variables, couplings):
    """Check that total_derivatives() returns a non-empty nested dictionary."""
    jac = mda.assembly.total_derivatives(
        in_data, None, variables, couplings, mode=JacobianAssembly.ADJOINT_MODE
    )
    assert jac["y_4"]["x_shared"] is None
    assert jac["y_1"]["TOTO"] is None


@pytest.mark.parametrize(
    "save,file_path,expected",
    [
        (False, None, None),
        (False, "foo", None),
        (True, None, "coupled_jacobian.pdf"),
        (True, "bar", "coupled_jacobian_bar.pdf"),
    ],
)
def test_plot_dependency_jacobian(mda, save, file_path, expected):
    """Check the file path used by plot_dependency_jacobian()."""
    with mock.patch.object(jacobian_assembly, "save_show_figure") as mock_method:
        assert (
            mda.assembly.plot_dependency_jacobian(
                ["y_4"], ["x_2"], filepath=file_path, save=save
            )
            == expected
        )

        if sys.version_info[:2] == (3, 7):
            args = mock_method.call_args[0]
        else:
            args = mock_method.call_args.args

        assert args[2] == expected


def test_lu_convergence_warning(assembly, caplog):
    random.seed(1)
    n_x = 5
    n_y = 10
    n_f = 1
    dres_dy_t = random.rand(n_y, n_y)
    dres_dy_t[0, :] = 0.0
    dres_dy_t[0, 0] = 1e-30
    dfun_dy = {"y_4": csr_matrix(random.rand(n_f, n_y))}
    dfun_dx = {"y_4": csr_matrix(random.rand(n_f, n_x))}
    dres_dx = csr_matrix(random.rand(n_y, n_x))

    assembly.coupled_system._adjoint_mode_lu(
        ["y_4"], dres_dx=dres_dx, dres_dy_t=dres_dy_t, dfun_dx=dfun_dx, dfun_dy=dfun_dy
    )

    expected = (
        "The linear system in _adjoint_mode_lu used to compute the coupled "
        "derivatives is not well resolved, residuals > tolerance"
    )

    assert expected in caplog.text

    assembly.coupled_system._direct_mode_lu(
        ["y_4"],
        n_variables=n_x,
        n_couplings=n_y,
        dres_dx=dres_dx,
        dres_dy=dres_dy_t,
        dfun_dx=dfun_dx,
        dfun_dy=dfun_dy,
    )

    expected = (
        "The linear system in _direct_mode_lu used to compute the coupled "
        "derivatives is not well resolved, residuals > tolerance"
    )

    assert expected in caplog.text
