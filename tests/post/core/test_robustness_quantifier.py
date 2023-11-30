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

from typing import TYPE_CHECKING

import pytest
from numpy import arange
from numpy import array
from numpy import eye
from numpy import ones
from numpy.linalg import norm
from numpy.random import default_rng
from scipy.optimize import rosen
from scipy.optimize import rosen_der

from gemseo import SEED
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.post.core.robustness_quantifier import RobustnessQuantifier
from gemseo.problems.analytical.rosenbrock import Rosenbrock

if TYPE_CHECKING:
    from gemseo.algos.database import Database


@pytest.fixture()
def database() -> Database:
    """The database."""
    n = 2
    problem = Rosenbrock(n)
    problem.x_0 = 1.0 - 2 * arange(n) / float(n)
    OptimizersFactory().execute(problem, "L-BFGS-B", max_iter=200)
    return problem.database


def test_init():
    """"""
    RobustnessQuantifier(None)


@pytest.mark.parametrize("args", [(), ("BFGS",), ("LEAST_SQUARES",)])
def test_init_methods(database, args):
    """"""
    RobustnessQuantifier(database, *args)


def test_build_approx(database):
    """"""
    for method in RobustnessQuantifier.Approximation:
        rq = RobustnessQuantifier(database, method)
        rq.compute_approximation(funcname="rosen", last_iter=-1)


def test_function_error(database):
    """"""
    n = 2
    rq = RobustnessQuantifier(database)
    rq.compute_approximation(funcname="rosen", last_iter=-1)
    rq.b_mat = None
    with pytest.raises(ValueError):
        rq.compute_function_approximation(ones(n))
    x = ones(n) + (array(list(range(n))) + 1) / (10.0 + n)
    with pytest.raises(ValueError):
        rq.compute_gradient_approximation(x)


@pytest.mark.parametrize("method", ["SR1", "BFGS"])
def test_approximation_precision(database, method):
    """"""
    n = 2
    rq = RobustnessQuantifier(database, method)
    rq.compute_approximation(funcname="rosen", first_iter=0, last_iter=-1)
    out = rq.compute_function_approximation(x_vars=ones(n))
    assert abs(out) < 1e-8
    x = 0.99 * ones(n)
    out = rq.compute_function_approximation(x)
    assert abs(out - rosen(x)) < 0.01
    outg = rq.compute_gradient_approximation(x)
    assert norm(outg - rosen_der(x)) / norm(rosen_der(x)) < 0.15
    x = ones(n) + (array(list(range(n))) + 1) / (10.0 + n)
    out = rq.compute_function_approximation(x)
    assert abs(out - rosen(x)) < 0.04
    outg = rq.compute_gradient_approximation(x)


def test_mc_average(database):
    """"""
    rq = RobustnessQuantifier(database)
    rq.compute_approximation(funcname="rosen")
    mu = ones(2)
    cov = 0.0001 * eye(2)
    rq.montecarlo_average_var(mu, cov)

    cov = 0.0001 * eye(3)
    with pytest.raises(ValueError):
        rq.montecarlo_average_var(mu, cov)


def test_compute_expected_value(database):
    """"""
    rq = RobustnessQuantifier(database)
    rq.compute_approximation(funcname="rosen")
    mu = ones(2)
    cov = 0.0001 * eye(2)
    e = rq.compute_expected_value(mu, cov)
    var = rq.compute_variance(mu, cov)
    assert e == pytest.approx(0.0501, abs=1e-4)
    assert var == pytest.approx(0.0050, abs=1e-4)

    e_ref, var_ref = rq.montecarlo_average_var(mu, cov, func=rosen, n_samples=300000)
    input_samples = default_rng(SEED).multivariate_normal(mu, cov, 300000).T
    output_samples = rosen(input_samples)
    assert output_samples.mean() == e_ref
    assert output_samples.var() == var_ref

    cov = 0.0001 * eye(3)
    with pytest.raises(ValueError):
        rq.compute_expected_value(mu, cov)

    cov = 0.0001 * eye(2)
    rq.b_mat = None
    with pytest.raises(ValueError):
        rq.compute_expected_value(mu, cov)


def test_compute_variance_error(database):
    """"""
    rq = RobustnessQuantifier(database)
    rq.compute_approximation(funcname="rosen")
    mu = ones(2)
    cov = 0.0001 * eye(3)
    with pytest.raises(ValueError):
        rq.compute_variance(mu, cov)

    cov = 0.0001 * eye(2)
    rq.b_mat = None
    with pytest.raises(ValueError):
        rq.compute_variance(mu, cov)
