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
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np
import pytest
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.post.core.hessians import BFGSApprox
from gemseo.post.core.hessians import HessianApproximation
from gemseo.post.core.hessians import LSTSQApprox
from gemseo.post.core.hessians import SR1Approx
from numpy import array
from numpy import multiply
from numpy import ndarray
from numpy import outer
from numpy.linalg import LinAlgError
from numpy.linalg import norm
from scipy.optimize import rosen_hess

MDF_HIST_PATH = Path(__file__).parent / "mdf_history.h5"
ROSENBROCK_2_PATH = Path(__file__).parent / "rosenbrock_2_opt_pb.h5"
ROSENBROCK_5_PATH = Path(__file__).parent / "rosenbrock_5_opt_pb.h5"
ROSENBROCK_35_PATH = Path(__file__).parent / "rosenbrock_35_opt_pb.h5"
ROSENBROCK_2_LB_UB_PATH = Path(__file__).parent / "rosenbrock_2_lb_ub_opt_pb.h5"


def build_history(
    problem_path: Path | str,
) -> tuple[ndarray, OptimizationResult, OptimizationProblem]:
    """Get the history of a Rosenbrock problem from an hdf file path.

    Args:
        problem_path: The file path to the selected case.

    Returns:
        The Rosenbrock hessian, the problem solution and the opt problem.
    """
    problem = OptimizationProblem.import_hdf(problem_path)
    x_opt = problem.solution.x_opt
    h_ref = rosen_hess(x_opt)
    return h_ref, problem.solution, problem


def compare_approximations(
    h_ref: ndarray,
    approx_class: HessianApproximation,
    problem: OptimizationProblem,
    ermax: float = 0.7,
    **kwargs: Mapping[str, Any],
) -> None:
    """Check that the approximated hessian is close enough to the reference one.

    Args:
        h_ref: The reference hessian.
        approx_class: The approximation class.
        ermax:  The maximum error.
        **kwargs: The approximation options.
    """
    database = problem.database
    n = problem.dimension
    approx = approx_class(database)
    h_approx, _, _, _ = approx.build_approximation(
        funcname=problem.objective.name, **kwargs
    )
    assert h_approx.shape == (n, n)
    error = compute_error(h_ref, h_approx)
    assert error <= ermax


def test_scaling():
    """Test that the scaling option works properly."""
    _, result, problem = build_history(ROSENBROCK_2_LB_UB_PATH)

    database = problem.database
    approx = HessianApproximation(database)
    design_space = problem.design_space
    _, _, _, _ = approx.build_approximation(
        funcname=problem.objective.name, scaling=True, design_space=design_space
    )

    approx = SR1Approx(database)
    design_space = problem.design_space
    h_approx_unscaled, _, _, _ = approx.build_approximation(
        funcname=problem.objective.name, scaling=True, design_space=design_space
    )

    h_approx_scaled, _, _, _ = approx.build_approximation(
        funcname=problem.objective.name,
        scaling=True,
        design_space=design_space,
        normalize_design_space=True,
    )

    h_exact = rosen_hess(result.x_opt)

    v = design_space._norm_factor
    scale_fact = outer(v, v.T)

    h_exact_scaled = multiply(h_exact, scale_fact)
    h_approx_unscaled_scaled = multiply(h_approx_unscaled, scale_fact)
    assert norm(h_exact_scaled - h_approx_unscaled_scaled) / norm(h_exact_scaled) < 1e-2
    assert norm(h_exact_scaled - h_approx_scaled) / norm(h_exact_scaled) < 1e-2


def compute_error(
    h_ref: ndarray,
    h_approx: ndarray,
) -> float:
    """Compute the error of the approximation.

    Args:
        h_ref: The reference hessian.
        h_approx: The approximated hessian.

    Returns:
        The error of the approximation.
    """
    return (norm(h_approx - h_ref) / norm(h_ref)) * 100


def test_baseclass_methods():
    """Test the different types of hessian approximation."""
    _, _, problem = build_history(ROSENBROCK_2_PATH)
    database = problem.database
    apprx = HessianApproximation(database)
    # 73 items in database
    at_most_niter = 2
    x_hist, x_grad_hist, n_iter, _ = apprx.get_x_grad_history(
        problem.objective.name, at_most_niter=at_most_niter
    )
    assert n_iter == at_most_niter
    assert x_hist.shape[0] == at_most_niter
    assert x_grad_hist.shape[0] == at_most_niter

    _, _, n_iter_ref, nparam = apprx.get_x_grad_history(problem.objective.name)

    _, _, n_iter_2, _ = apprx.get_x_grad_history(
        problem.objective.name, last_iter=n_iter_ref
    )

    assert n_iter_ref == n_iter_2
    _, _, n_iter_3, _ = apprx.get_x_grad_history(problem.objective.name, first_iter=10)

    assert n_iter_ref == n_iter_3 + 10

    apprx.build_approximation(
        problem.objective.name, b_mat0=np.eye(nparam), save_matrix=True
    )

    assert len(apprx.b_mat_history) > 1

    with (pytest.raises(ValueError)):
        apprx.get_x_grad_history(problem.objective.name, at_most_niter=1)
    database.clear()

    with (pytest.raises(ValueError)):
        apprx.get_x_grad_history(problem.objective.name, at_most_niter=at_most_niter)

    with (pytest.raises(ValueError)):
        apprx.get_x_grad_history(
            problem.objective.name,
            at_most_niter=at_most_niter,
            normalize_design_space=True,
        )


def test_get_x_grad_history_on_sobieski():
    """Test the gradient history on the Sobieski problem."""
    opt_pb = OptimizationProblem.import_hdf(MDF_HIST_PATH)
    apprx = HessianApproximation(opt_pb.database)
    with (pytest.raises(ValueError)):
        apprx.get_x_grad_history("g_1")
    x_hist, x_grad_hist, n_iter, nparam = apprx.get_x_grad_history("g_1", func_index=1)

    assert len(x_hist) == 4
    assert n_iter == 4
    assert nparam == 10
    for x in x_hist:
        assert x.shape == (nparam,)

    assert len(x_hist) == len(x_grad_hist)
    for grad in x_grad_hist:
        assert grad.shape == (nparam,)

    with pytest.raises(ValueError):
        apprx.get_s_k_y_k(x_hist, x_grad_hist, 5)

    with (pytest.raises(ValueError)):
        apprx.get_x_grad_history("g_1", func_index=7)

    # Create inconsistent optimization history by restricting g_2 gradient
    # size
    x_0 = next(iter(opt_pb.database.keys()))
    val_0 = opt_pb.database[x_0]
    val_0["@g_2"] = val_0["@g_2"][1:]
    with pytest.raises(ValueError):
        apprx.get_x_grad_history("g_2")


def test_n_2():
    """Test the hessian approximation with the Rosenbrock problem at n=2."""
    h_ref, _, problem = build_history(ROSENBROCK_2_PATH)
    compare_approximations(h_ref, BFGSApprox, problem, first_iter=8, ermax=3.0)
    compare_approximations(h_ref, LSTSQApprox, problem, first_iter=13, ermax=20.0)
    compare_approximations(h_ref, SR1Approx, problem, first_iter=7, ermax=30.0)
    compare_approximations(
        h_ref, HessianApproximation, problem, first_iter=8, ermax=30.0
    )


def test_n_5():
    """Test the hessian approximation with the Rosenbrock problem at n=5."""
    h_ref, _, problem = build_history(ROSENBROCK_5_PATH)
    compare_approximations(h_ref, BFGSApprox, problem, first_iter=5, ermax=30.0)
    compare_approximations(h_ref, LSTSQApprox, problem, first_iter=19, ermax=40.0)
    compare_approximations(h_ref, SR1Approx, problem, first_iter=5, ermax=30.0)
    compare_approximations(
        h_ref, HessianApproximation, problem, first_iter=5, ermax=30.0
    )


def test_n_35():
    """Test the hessian approximation with the Rosenbrock problem at n=35."""
    h_ref, _, problem = build_history(ROSENBROCK_35_PATH)
    compare_approximations(h_ref, SR1Approx, problem, first_iter=5, ermax=40.0)
    compare_approximations(h_ref, LSTSQApprox, problem, first_iter=30, ermax=110.0)
    compare_approximations(h_ref, SR1Approx, problem, first_iter=30, ermax=47.0)
    compare_approximations(
        h_ref, HessianApproximation, problem, first_iter=45, ermax=72.0
    )


def test_build_inverse_approximation():
    """Test the creation of an inverse approximation."""
    _, _, problem = build_history(ROSENBROCK_2_LB_UB_PATH)
    database = problem.database
    approx = HessianApproximation(database)
    funcname = problem.objective.name
    approx.build_inverse_approximation(funcname=funcname, h_mat0=[], factorize=True)
    with pytest.raises(LinAlgError):
        approx.build_inverse_approximation(funcname=funcname, h_mat0=array([1.0, 2.0]))
    with pytest.raises(LinAlgError):
        approx.build_inverse_approximation(
            funcname=funcname,
            h_mat0=array([[0.0, 1.0], [-1.0, 0.0]]),
            factorize=True,
        )
    approx.build_inverse_approximation(
        funcname=funcname, h_mat0=array([[1.0, 0.0], [0.0, 1.0]]), factorize=True
    )
    approx.build_inverse_approximation(funcname=funcname, return_x_grad=True)
    x_hist, x_grad_hist, _, _ = approx.get_x_grad_history(problem.objective.name)
    x_corr, grad_corr = approx.compute_corrections(x_hist, x_grad_hist)
    approx.rebuild_history(x_corr, x_hist[0], grad_corr, x_grad_hist[0])
