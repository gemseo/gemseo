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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re

import pytest
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mda.mda_chain import MDAChain
from gemseo.mda.newton import MDANewtonRaphson
from gemseo.mda.newton import MDAQuasiNewton
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sellar.sellar import X_SHARED
from gemseo.problems.sellar.sellar import Y_1
from gemseo.problems.sellar.sellar import Y_2
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from numpy import array
from numpy import float64
from numpy import isclose
from numpy import linalg
from numpy import ones

from .test_gauss_seidel import SelfCoupledDisc

TRESHOLD_MDA_TOL = 1e-6


@pytest.mark.parametrize("coupl_scaling", [True, False])
def test_raphson_sobieski(coupl_scaling):
    """Test the execution of Gauss-Seidel on Sobieski."""
    disciplines = [
        SobieskiAerodynamics(),
        SobieskiStructure(),
        SobieskiPropulsion(),
        SobieskiMission(),
    ]
    mda = MDANewtonRaphson(disciplines)
    mda.set_residuals_scaling_options(scale_residuals_with_coupling_size=coupl_scaling)
    mda.matrix_type = JacobianAssembly.SPARSE
    mda.reset_history_each_run = True
    mda.execute()
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL

    mda.warm_start = True
    mda.execute({"x_1": mda.default_inputs["x_1"] + 1.0e-2})
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL


@pytest.mark.parametrize("relax_factor", [-0.1, 1.1])
def test_newton_raphson_invalid_relax_factor(relax_factor):
    expected = re.escape(
        "Newton relaxation factor should belong to (0, 1] (current value: {}).".format(
            relax_factor
        )
    )
    with pytest.raises(ValueError, match=expected):
        MDANewtonRaphson([Sellar1(), Sellar2()], relax_factor=relax_factor)


def get_sellar_initial():
    """Generate initial solution."""
    x_local = array([0.0], dtype=float64)
    x_shared = array([1.0, 0.0], dtype=float64)
    y_0 = ones(1, dtype=float64)
    y_1 = ones(1, dtype=float64)
    return x_local, x_shared, y_0, y_1


def test_raphson_sobieski_sparse():
    """Test the execution of Gauss-Seidel on Sobieski."""
    disciplines = [
        SobieskiAerodynamics(),
        SobieskiStructure(),
        SobieskiPropulsion(),
        SobieskiMission(),
    ]
    mda = MDANewtonRaphson(disciplines)
    mda.matrix_type = JacobianAssembly.LINEAR_OPERATOR
    mda.execute()
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL


def test_quasi_newton_invalida_method():
    with pytest.raises(
        ValueError, match="Method 'unknown_method' is not a valid quasi-Newton method."
    ):
        MDAQuasiNewton([Sellar1(), Sellar2()], method="unknown_method")


def test_wrong_name():
    disciplines = [
        SobieskiAerodynamics(),
        SobieskiStructure(),
        SobieskiPropulsion(),
        SobieskiMission(),
    ]
    with pytest.raises(ValueError, match="is not a valid quasi-Newton method"):
        MDAQuasiNewton(disciplines, method="FAIL")


def test_raphson_sellar_sparse_complex():
    disciplines = [Sellar1(), Sellar2()]
    mda = MDANewtonRaphson(disciplines)
    mda.matrix_type = JacobianAssembly.SPARSE
    mda.execute()

    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL

    y_ref = array([0.80004953, 1.79981434])
    y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
    assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-4


@pytest.mark.parametrize("use_cache", [True, False])
def test_raphson_sellar_without_cache(use_cache):
    """Test the execution of Newton on Sellar case.

    This test also checks that each Newton step implies one disciplinary call, and one
    disciplinary linearization, whatever a cache mechanism is used or not.
    """
    disciplines = [Sellar1(), Sellar2()]
    if not use_cache:
        for disc in disciplines:
            disc.cache = None
    mda = MDANewtonRaphson(disciplines)
    mda.execute()

    residual_length = len(mda.residual_history)
    assert mda.residual_history[-1] < 1e-6
    assert disciplines[0].n_calls == residual_length
    assert disciplines[0].n_calls_linearize == residual_length


def test_raphson_sellar():
    """Test the execution of Newton on Sobieski."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDANewtonRaphson(disciplines)
    mda.execute()

    assert mda.residual_history[-1] < 1e-6

    y_ref = array([0.80004953, 1.79981434])
    y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
    assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-4


def test_raphson_sellar_linop():
    disciplines = [Sellar1(), Sellar2()]
    mda = MDANewtonRaphson(disciplines)
    mda.matrix_type = JacobianAssembly.LINEAR_OPERATOR
    mda.execute()
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL


def test_broyden_sellar():
    """Test the execution of quasi-Newton on Sellar."""
    mda = MDAQuasiNewton([Sellar1(), Sellar2()], method=MDAQuasiNewton.BROYDEN1)
    mda.reset_history_each_run = True
    mda.execute()
    assert mda.residual_history[-1] < 1e-5

    y_ref = array([0.80004953, 1.79981434])
    y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
    assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-3

    mda.warm_start = True
    mda.execute({X_SHARED: mda.default_inputs[X_SHARED] + 0.1})


def test_hybrid_sellar():
    """Test the execution of quasi-Newton on Sellar."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDAQuasiNewton(disciplines, use_gradient=True)

    mda.execute()

    y_ref = array([0.80004953, 1.79981434])
    y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
    assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-4


def test_lm_sellar():
    """Test the execution of quasi-Newton on Sellar."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDAQuasiNewton(
        disciplines, method=MDAQuasiNewton.LEVENBERG_MARQUARDT, use_gradient=True
    )
    mda.execute()

    y_ref = array([0.80004953, 1.79981434])
    y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
    assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-4


def test_dfsane_sellar():
    """Test the execution of quasi-Newton on Sellar."""
    mda = MDAQuasiNewton([Sellar1(), Sellar2()], method=MDAQuasiNewton.DF_SANE)
    mda.execute()

    y_ref = array([0.80004953, 1.79981434])
    y_opt = array([mda.local_data[Y_1][0].real, mda.local_data[Y_2][0].real])
    assert linalg.norm(y_ref - y_opt) / linalg.norm(y_ref) < 1e-3

    """Test the execution of quasi-Newton with fake method."""
    with pytest.raises(
        ValueError, match="Method 'unknown_method' is not a valid quasi-Newton method."
    ):
        MDAQuasiNewton([Sellar1(), Sellar2()], method="unknown_method")


def test_broyden_sellar2():
    """Test the execution of quasi-Newton on Sellar."""
    disciplines = [Sellar1(), SellarSystem()]
    mda = MDAQuasiNewton(disciplines, method=MDAQuasiNewton.BROYDEN1)
    mda.reset_history_each_run = True
    mda.execute()

    assert mda.local_data[mda.RESIDUALS_NORM][0] < 1e-6


def test_self_coupled():
    sc_disc = SelfCoupledDisc()
    mda = MDAQuasiNewton([sc_disc], tolerance=1e-14, max_mda_iter=40)
    out = mda.execute()
    assert abs(out["y"] - 2.0 / 3.0) < 1e-6


def test_log_convergence():
    """Check that the boolean log_convergence is correctly set."""
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    mda = MDANewtonRaphson(disciplines)
    assert not mda.log_convergence
    mda = MDANewtonRaphson(disciplines, log_convergence=True)
    assert mda.log_convergence


@pytest.mark.parametrize(
    "mda_class,expected_obj",
    [("MDAQuasiNewton", -591.35), ("MDANewtonRaphson", -608.175)],
)
def test_parallel_doe(mda_class, expected_obj, generate_parallel_doe_data):
    """Test the execution of Newton methods in parallel.

    Args:
        mda_class: The specific Newton MDA to test.
        expected_obj: The expected objective value of the DOE scenario.
        generate_parallel_doe_data: Fixture that returns the optimum solution to
            a parallel DOE scenario for a particular `main_mda_name`.
    """
    obj = generate_parallel_doe_data(mda_class)
    assert isclose(array([obj]), array([expected_obj]), atol=1e-3)


def test_weak_and_strong_couplings():
    """Test the Newton method on a simple Analytic case with strong and weak
    couplings."""
    disc1 = AnalyticDiscipline(expressions={"z": "2*x"})
    disc2 = AnalyticDiscipline(expressions={"i": "z + j"})
    disc3 = AnalyticDiscipline(expressions={"j": "1 - 0.3*i"})
    disc4 = AnalyticDiscipline(expressions={"obj": "i+j"})
    disciplines = [disc1, disc2, disc3, disc4]
    mda = MDANewtonRaphson(
        disciplines,
    )
    mda.execute(
        {"z": array([0.0]), "i": array([0.0]), "j": array([0.0]), "x": array([0.0])}
    )
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL
    assert mda.local_data[mda.RESIDUALS_NORM][0] < TRESHOLD_MDA_TOL
    assert mda.local_data["obj"] == pytest.approx(array([2.0 / 1.3]))


def test_weak_and_strong_couplings_two_cycles():
    """Test the Newton method on a simple Analytic case.

    Two strongly coupled cycles of disciplines are used in this test case.
    """
    disc1 = AnalyticDiscipline(expressions={"z": "2*x"})
    disc2 = AnalyticDiscipline(expressions={"i": "z + 0.2*j"})
    disc3 = AnalyticDiscipline(expressions={"j": "1. - 0.3*i"})
    disc4 = AnalyticDiscipline(expressions={"k": "i+j"})
    disc5 = AnalyticDiscipline(expressions={"l": "k + 0.2*m"})
    disc6 = AnalyticDiscipline(expressions={"m": "1. - 0.3*l"})
    disc7 = AnalyticDiscipline(expressions={"obj": "l+m"})
    disciplines = [disc1, disc2, disc3, disc4, disc5, disc6, disc7]
    mda = MDANewtonRaphson(
        disciplines,
    )
    mda.linearization_mode = "adjoint"
    mda.add_differentiated_inputs(["x"])
    mda.add_differentiated_outputs(["obj"])
    mda_input = {
        "z": array([1.0]),
        "i": array([0.0]),
        "j": array([0.0]),
        "k": array([0.0]),
        "l": array([0.0]),
        "x": array([0.0]),
    }
    out = mda.execute(mda_input)
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL

    mda_ref = MDAChain(disciplines)
    mda_ref.linearization_mode = "adjoint"
    mda_ref.add_differentiated_inputs(["x"])
    mda_ref.add_differentiated_outputs(["obj"])
    out_ref = mda_ref.execute(mda_input)

    for output_name in mda.get_output_data_names():
        if output_name == mda.RESIDUALS_NORM:
            continue
        assert out[output_name] == pytest.approx(out_ref[output_name])

    assert mda.check_jacobian(
        input_data=mda_input,
        inputs=["x"],
        outputs=["obj"],
        linearization_mode="adjoint",
        threshold=1e-3,
    )
