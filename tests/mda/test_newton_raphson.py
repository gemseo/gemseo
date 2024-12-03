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

import pickle
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from numpy import allclose as allclose_
from numpy import array
from numpy import linalg
from numpy.testing import assert_equal

from gemseo import create_mda
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mda.mda_chain import MDAChain
from gemseo.mda.newton_raphson import MDANewtonRaphson
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.problems.mdo.sellar.utils import get_y_opt
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline
    from gemseo.typing import StrKeyMapping

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline

TRESHOLD_MDA_TOL = 1e-6
SELLAR_Y_REF = array([0.80004953, 1.79981434])


def allclose(a, b):
    return allclose_(a, b, atol=1e-8, rtol=0.0)


@pytest.fixture(scope="module")
def mda_setting() -> StrKeyMapping:
    """Returns the setting for all subsequent MDAs."""
    return {"tolerance": 1e-12, "max_mda_iter": 50}


@pytest.fixture(scope="module")
def sobieski_disciplines() -> Sequence[Discipline]:
    """Returns the Sobieski's disciplines."""
    return [SobieskiAerodynamics(), SobieskiStructure(), SobieskiPropulsion()]


@pytest.fixture(scope="module")
def reference(sobieski_disciplines, mda_setting) -> MDANewtonRaphson:
    """An instance of Newton-Raphson MDA on the Sobieski problem."""
    mda_newton = MDANewtonRaphson(
        sobieski_disciplines, **mda_setting, over_relaxation_factor=1.0
    )
    mda_newton.execute()
    return mda_newton


@pytest.mark.parametrize("relaxation", [0.8, 1.0, 1.2])
def test_over_relaxation(
    sobieski_disciplines, mda_setting, relaxation, reference
) -> None:
    """Tests the relaxation factor."""
    mda = MDANewtonRaphson(
        sobieski_disciplines, **mda_setting, over_relaxation_factor=relaxation
    )
    mda.execute()

    assert allclose(
        mda.get_current_resolved_residual_vector(),
        reference.get_current_resolved_residual_vector(),
    )
    assert allclose(
        mda.get_current_resolved_variables_vector(),
        reference.get_current_resolved_variables_vector(),
    )


@pytest.mark.parametrize("acceleration", AccelerationMethod)
def test_acceleration_methods(
    sobieski_disciplines, mda_setting, acceleration, reference
) -> None:
    """Tests the acceleration methods."""
    mda = MDANewtonRaphson(
        sobieski_disciplines,
        **mda_setting,
        acceleration_method=acceleration,
        over_relaxation_factor=1.0,
    )
    mda.execute()

    assert mda._current_iter <= reference._current_iter

    assert allclose(
        mda.get_current_resolved_residual_vector(),
        reference.get_current_resolved_residual_vector(),
    )
    assert allclose(
        mda.get_current_resolved_variables_vector(),
        reference.get_current_resolved_variables_vector(),
    )


@pytest.mark.parametrize("coupl_scaling", ["n_coupling_variables", "no_scaling"])
def test_raphson_sobieski(coupl_scaling) -> None:
    """Test the execution of Gauss-Seidel on Sobieski."""
    disciplines = [SobieskiAerodynamics(), SobieskiStructure(), SobieskiPropulsion()]
    mda = MDANewtonRaphson(disciplines)
    mda.scaling = coupl_scaling
    mda.matrix_type = JacobianAssembly.JacobianType.MATRIX
    mda.reset_history_each_run = True
    mda.execute()
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL

    mda.settings.warm_start = True
    mda.execute({"x_1": mda.io.input_grammar.defaults["x_1"] + 1.0e-2})
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL


def test_raphson_sobieski_sparse() -> None:
    """Test the execution of Newton-Raphson MDA on Sobieski."""
    disciplines = [SobieskiAerodynamics(), SobieskiStructure(), SobieskiPropulsion()]
    mda = MDANewtonRaphson(disciplines)
    mda.matrix_type = JacobianAssembly.JacobianType.LINEAR_OPERATOR
    mda.execute()
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL


def test_raphson_sellar_sparse_complex() -> None:
    """Test the execution of Newton-Raphson MDA on Sellar with complex numbers."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDANewtonRaphson(disciplines)
    mda.matrix_type = JacobianAssembly.JacobianType.MATRIX
    mda.execute()

    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL

    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-4


@pytest.mark.parametrize("use_cache", [True, False])
def test_raphson_sellar_without_cache(use_cache) -> None:
    """Test the execution of Newton on Sellar case.

    This test also checks that each Newton step implies one disciplinary call, and one
    disciplinary linearization, whatever a cache mechanism is used or not.
    """
    disciplines = [Sellar1(), Sellar2()]
    if not use_cache:
        for disc in disciplines:
            disc.cache = None
    tolerance = 1e-12
    mda = MDANewtonRaphson(disciplines, tolerance=tolerance)
    mda.execute()

    residual_length = len(mda.residual_history)
    assert mda.residual_history[-1] < tolerance
    assert disciplines[0].execution_statistics.n_executions == residual_length
    assert disciplines[0].execution_statistics.n_linearizations == residual_length - 1


@pytest.mark.parametrize("parallel", [False, True])
def test_raphson_sellar(parallel) -> None:
    """Test the execution of Newton on Sobieski."""
    disciplines = [Sellar1(), Sellar2()]
    kwargs = {"n_processes": 2} if parallel else {"n_processes": 1}
    mda = MDANewtonRaphson(disciplines, **kwargs)
    mda.execute()

    assert (mda.settings.n_processes == 1) is not parallel
    assert mda.residual_history[-1] < 1e-6
    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-4


def test_sellar_linop() -> None:
    """Test residuals jacobian as a linear operator."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDANewtonRaphson(disciplines)
    mda.matrix_type = JacobianAssembly.JacobianType.LINEAR_OPERATOR
    mda.execute()
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL


def test_log_convergence() -> None:
    """Check that the boolean log_convergence is correctly set."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDANewtonRaphson(disciplines)
    assert not mda.settings.log_convergence
    mda = MDANewtonRaphson(disciplines, log_convergence=True)
    assert mda.settings.log_convergence


def test_weak_and_strong_couplings() -> None:
    """Test the Newton method on a simple Analytic case with strong and weak
    couplings."""
    disc1 = AnalyticDiscipline({"z": "2*x"}, name="z=f(x) disc")
    disc2 = AnalyticDiscipline({"i": "z + j"}, name="i=f(z,j) disc")
    disc3 = AnalyticDiscipline({"j": "1 - 0.3*i"}, name="j=f(i) disc")
    disc4 = AnalyticDiscipline({"obj": "i+j"}, name="obj=f(i,j) disc")
    disciplines = [disc1, disc2, disc3, disc4]
    mda = MDAChain(disciplines, inner_mda_name="MDANewtonRaphson")
    mda.execute({
        "z": array([0.0]),
        "i": array([0.0]),
        "j": array([0.0]),
        "x": array([0.0]),
    })
    assert mda.inner_mdas[0].residual_history[-1] < TRESHOLD_MDA_TOL
    assert mda.io.data[mda.NORMALIZED_RESIDUAL_NORM][0] < TRESHOLD_MDA_TOL
    assert mda.io.data["obj"] == pytest.approx(array([2.0 / 1.3]))


def test_weak_and_strong_couplings_two_cycles() -> None:
    """Test the Newton method on a simple Analytic case.

    Two strongly coupled cycles of disciplines are used in this test case.
    """
    disc1 = AnalyticDiscipline({"z": "2*x"}, name=1)
    disc2 = AnalyticDiscipline({"i": "z + 0.2*j"}, name=2)
    disc3 = AnalyticDiscipline({"j": "1. - 0.3*i"}, name=3)
    disc4 = AnalyticDiscipline({"k": "i+j"}, name=4)
    disc5 = AnalyticDiscipline({"l": "k + 0.2*m"}, name=5)
    disc6 = AnalyticDiscipline({"m": "1. - 0.3*l"}, name=6)
    disc7 = AnalyticDiscipline({"obj": "l+m"}, name=7)
    disciplines = [disc1, disc2, disc3, disc4, disc5, disc6, disc7]
    mda = MDAChain(disciplines, inner_mda_name="MDANewtonRaphson", tolerance=1e-13)
    mda.settings.warm_start = True
    mda.linearization_mode = "adjoint"
    mda_input = {
        "z": array([1.0]),
        "i": array([0.0]),
        "j": array([0.0]),
        "k": array([0.0]),
        "l": array([0.0]),
        "x": array([0.0]),
    }
    out = mda.execute(mda_input)
    for mda_i in mda.inner_mdas:
        assert mda_i.residual_history[-1] < TRESHOLD_MDA_TOL

    mda_ref = MDAChain(disciplines)
    out_ref = mda_ref.execute(mda_input)

    for output_name in mda.io.output_grammar:
        if output_name == mda.NORMALIZED_RESIDUAL_NORM:
            continue
        assert out[output_name] == pytest.approx(out_ref[output_name], rel=1e-5)

    assert mda.check_jacobian(
        input_data=mda_input,
        input_names=["x"],
        output_names=["obj"],
        linearization_mode="adjoint",
        threshold=1e-3,
        step=1e-4,
    )


@pytest.mark.parametrize(
    (
        "mda_linear_solver",
        "mda_linear_solver_settings",
        "newton_linear_solver_name",
        "newton_linear_solver_settings",
    ),
    [
        ("DEFAULT", {}, "DEFAULT", {}),
        ("DEFAULT", {"atol": 1e-6}, "DEFAULT", {}),
        ("DEFAULT", {}, "DEFAULT", {"atol": 1e-3}),
        ("BICG", {}, "DEFAULT", {}),
        ("DEFAULT", {}, "BICG", {}),
    ],
)
def test_pass_dedicated_newton_options(
    mda_linear_solver,
    mda_linear_solver_settings,
    newton_linear_solver_name,
    newton_linear_solver_settings,
) -> None:
    """Test that the linear solver type and options for the Adjoint method and the
    newton method can be controlled independently in a newton based MDA. A mock is used
    to unitary test the arguments passed to the Newton step.

    Args:
        mda_linear_solver: The linear solver name to solve the MDA Adjoint matrix.
        mda_linear_solver_settings: The options for MDA matrix linear solver.
        newton_linear_solver_name: The linear solver name to solve the Newton method.
        newton_linear_solver_settings: The options for Newton linear solver.

    Returns:
    """
    # newton_linear_solver_settings = {"atol": 1e-6}
    mda = create_mda(
        "MDANewtonRaphson",
        disciplines=[Sellar1(), Sellar2()],
        linear_solver=mda_linear_solver,
        linear_solver_settings=mda_linear_solver_settings,
        newton_linear_solver_name=newton_linear_solver_name,
        newton_linear_solver_settings=newton_linear_solver_settings,
    )
    mda.assembly.compute_newton_step = mock.Mock(
        return_value=(array([-0.1935616 + 0.0j, 0.7964384 + 0.0j]), True)
    )
    mda.execute()
    newton_step_args = mda.assembly.compute_newton_step.call_args
    assert mda.settings.linear_solver == mda_linear_solver
    if mda_linear_solver_settings is None:
        assert mda.settings.linear_solver_settings == {}
    else:
        assert mda.settings.linear_solver_settings == mda_linear_solver_settings
    assert newton_step_args.args[2] == newton_linear_solver_name
    del newton_step_args.kwargs["matrix_type"]
    if "atol" in newton_linear_solver_settings:
        assert newton_step_args.kwargs["atol"] == newton_linear_solver_settings["atol"]


@pytest.mark.parametrize(
    ("newton_linear_solver_name", "newton_linear_solver_settings"),
    [
        ("DEFAULT", {"atol": 1e-7}),
        ("DEFAULT", {}),
        ("BICGSTAB", {}),
        ("GMRES", {}),
    ],
)
def test_mda_newton_convergence_passing_dedicated_newton_options(
    newton_linear_solver_name,
    newton_linear_solver_settings,
) -> None:
    """Test that Newton MDA converges toward expected value for various linear solver
    algorithms for the Newton method.

    Args:
        newton_linear_solver_name: The linear solver name to solve the Newton method.
        newton_linear_solver_settings: The options for Newton linear solver.

    Returns:
    """
    mda = create_mda(
        "MDANewtonRaphson",
        disciplines=[Sellar1(), Sellar2()],
        newton_linear_solver_name=newton_linear_solver_name,
        newton_linear_solver_settings=newton_linear_solver_settings,
    )
    mda.execute()
    assert mda.residual_history[-1] < TRESHOLD_MDA_TOL
    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-4


def test_mda_newton_serialization(tmp_wd) -> None:
    """Test serialization and deserialization of a Newton based MDA."""
    options = {"atol": 1e-6}
    mda = create_mda(
        "MDANewtonRaphson",
        disciplines=[Sellar1(), Sellar2()],
        newton_linear_solver_settings=options,
    )
    out = mda.execute()
    out_file = "mda_newton.pkl"
    with open(out_file, "wb") as file:
        pickle.dump(mda, file)

    with open(out_file, "rb") as file:
        mda_d = pickle.load(file)

    assert_equal(mda_d.io.data, out)


def test_mda_newton_weak_couplings() -> None:
    """Test the check when there are weakly coupled disciplines."""
    match = (
        "The MDANewtonRaphson has weakly coupled disciplines, which is not supported."
    )

    with pytest.raises(ValueError, match=match):
        create_mda("MDANewtonRaphson", disciplines=[Sellar1(), SellarSystem()])


def test_mda_newton_no_couplings() -> None:
    """Test the check when there are no coupled disciplines."""
    match = "There is no couplings to compute. Please consider using MDAChain."

    with pytest.raises(ValueError, match=match):
        create_mda("MDANewtonRaphson", disciplines=[Sellar1()])


def test_linear_solver_not_converged(caplog) -> None:
    """Test the warning message when the linear solver does not converge."""
    solver = "LGMRES"
    mda = create_mda(
        "MDANewtonRaphson",
        disciplines=[Sellar1(), Sellar2()],
        newton_linear_solver_name=solver,
        newton_linear_solver_settings={"max_iter": 1},
        max_mda_iter=2,
    )
    expected_log = (
        f"The linear solver {solver} failed to converge"
        " during the Newton's step computation."
    )
    mda.execute()
    assert expected_log in caplog.text
