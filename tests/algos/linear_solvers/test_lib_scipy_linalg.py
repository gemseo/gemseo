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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import logging
import pickle
import re
from os import remove
from pathlib import Path

import pytest
from numpy import eye
from numpy import ones
from numpy import zeros
from numpy.random import default_rng
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spilu

from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from gemseo.algos.linear_solvers.scipy_linalg.scipy_linalg import ScipyLinalgAlgos
from gemseo.algos.linear_solvers.scipy_linalg.settings.lgmres import LGMRES_Settings
from gemseo.utils.seeder import SEED

RESIDUALS_TOL = 1e-12


def test_algo_list() -> None:
    """Tests the algo list detection at lib creation."""
    factory = LinearSolverLibraryFactory()
    assert len(factory.algorithms) >= 6
    for algo in ("LGMRES", "GMRES", "BICG", "QMR", "BICGSTAB", "DEFAULT"):
        factory.is_available(algo)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 1000},
        {"settings_model": LGMRES_Settings(max_iter=1000)},
    ],
)
def test_default(kwargs) -> None:
    """Tests the DEFAULT solver."""
    factory = LinearSolverLibraryFactory()
    rng = default_rng(1)
    n = 5
    problem = LinearProblem(rng.random((n, n)), rng.random(n))
    factory.execute(problem, algo_name="DEFAULT", **kwargs)
    assert problem.solution is not None
    assert problem.compute_residuals() < RESIDUALS_TOL


@pytest.mark.parametrize("n", [1, 4, 20])
@pytest.mark.parametrize("algo_name", ["DEFAULT", "LGMRES", "BICGSTAB"])
@pytest.mark.parametrize("use_preconditioner", [True, False])
@pytest.mark.parametrize("use_ilu_precond", [True, False])
@pytest.mark.parametrize("use_x0", [True, False])
def test_linsolve(algo_name, n, use_preconditioner, use_x0, use_ilu_precond) -> None:
    """Tests the solvers options."""
    factory = LinearSolverLibraryFactory()
    rng = default_rng(1)
    problem = LinearProblem(rng.random((n, n)), rng.random(n))
    options = {
        "max_iter": 100,
        "rtol": 1e-14,
        "atol": 1e-13,
        "x0": None,
        "use_ilu_precond": use_ilu_precond,
    }
    if use_preconditioner and not use_ilu_precond:
        options["preconditioner"] = LinearOperator(
            problem.lhs.shape, spilu(problem.lhs).solve
        )
    if algo_name == "lgmres":
        v = rng.random(n)
        options.update({
            "inner_m": 10,
            "outer_k": 10,
            "outer_v": [(v, problem.lhs.dot(v))],
            "store_outer_av": True,
            "prepend_outer_v": True,
        })
    factory.execute(problem, algo_name=algo_name, **options)
    assert problem.solution is not None
    assert problem.compute_residuals() < RESIDUALS_TOL

    assert problem.solution is not None
    assert norm(problem.lhs.dot(problem.solution) - problem.rhs) < RESIDUALS_TOL


def test_common_dtype_cplx() -> None:
    factory = LinearSolverLibraryFactory()
    problem = LinearProblem(eye(2, dtype="complex128"), ones(2))
    factory.execute(problem, algo_name="DEFAULT")
    assert problem.compute_residuals() < RESIDUALS_TOL

    problem = LinearProblem(eye(2), ones(2, dtype="complex128"))
    factory.execute(problem, algo_name="DEFAULT")
    assert problem.compute_residuals() < RESIDUALS_TOL


def test_not_converged(caplog) -> None:
    """Tests the cases when convergence fails and save_when_fail option."""
    factory = LinearSolverLibraryFactory()
    rng = default_rng(SEED)
    n = 100
    problem = LinearProblem(rng.random((n, n)), rng.random(n))
    lib = factory.create("BICGSTAB")
    caplog.set_level(logging.WARNING)
    lib.execute(problem, max_iter=2, save_when_fail=True, use_ilu_precond=False)
    assert not problem.is_converged
    assert "The linear solver BICGSTAB did not converge." in caplog.text

    with Path(lib.file_path).open("rb") as f:
        problem2 = pickle.load(f)
    remove(lib.file_path)
    assert (problem2.lhs == problem.lhs).all()
    assert (problem2.rhs == problem.rhs).all()

    lib.execute(problem, max_iter=2, save_when_fail=True, use_ilu_precond=True)
    assert problem.is_converged


@pytest.mark.parametrize("seed", range(3))
def test_hard_conv(tmp_wd, seed) -> None:
    rng = default_rng(seed)
    n = 300
    problem = LinearProblem(rng.random((n, n)), rng.random(n))
    LinearSolverLibraryFactory().execute(
        problem,
        algo_name="DEFAULT",
        max_iter=3,
        store_residuals=True,
        use_ilu_precond=True,
        rtol=1e-14,
    )

    assert problem.compute_residuals() < 1e-10


def test_inconsistent_options() -> None:
    problem = LinearProblem(ones((2, 2)), ones(2))

    with pytest.raises(
        ValueError, match=re.escape("matrix and preconditioner have different shapes")
    ):
        LinearSolverLibraryFactory().execute(
            problem, algo_name="DEFAULT", preconditioner=ones((3, 3))
        )

    with pytest.raises(
        ValueError, match=re.escape("shapes of A (2, 2) and x0 (3,) are incompatible")
    ):
        LinearSolverLibraryFactory().execute(problem, algo_name="DEFAULT", x0=ones(3))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Use either 'use_ilu_precond' or provide 'preconditioner', but not both."
        ),
    ):
        LinearSolverLibraryFactory().execute(
            problem,
            algo_name="DEFAULT",
            preconditioner=ones((2, 2)),
            use_ilu_precond=True,
        )


def test_check_info() -> None:
    lib = ScipyLinalgAlgos("LGMRES")
    lib._problem = LinearProblem(zeros((2, 2)), ones(2))
    with pytest.raises(RuntimeError, match="illegal input or breakdown"):
        lib._check_solver_info(-1, {})


def test_factory() -> None:
    assert "ScipyLinalgAlgos" in LinearSolverLibraryFactory().linear_solvers
