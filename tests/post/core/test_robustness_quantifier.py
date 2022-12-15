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

import unittest

import numpy as np
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.post.core.robustness_quantifier import RobustnessQuantifier
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy.linalg import norm
from scipy.optimize import rosen
from scipy.optimize import rosen_der
from scipy.optimize import rosen_hess


class TestRobustnessQuantifier(unittest.TestCase):
    """"""

    @classmethod
    def setUpClass(cls):
        """"""
        cls.build_history(n=2)
        np.random.seed(1)

    @classmethod
    def build_history(cls, n):
        """

        :param n:

        """
        cls.n = n
        cls.problem = Rosenbrock(n)
        cls.problem.x_0 = 1.0 - 2 * np.arange(n) / float(n)
        cls.database = cls.problem.database
        cls.result = OptimizersFactory().execute(cls.problem, "L-BFGS-B", max_iter=200)

        cls.x_opt = cls.result.x_opt
        cls.H_ref = rosen_hess(cls.x_opt)

    def test_init(self):
        """"""
        RobustnessQuantifier(None)

    def test_method_error(self):
        """"""
        self.assertRaises(
            ValueError, RobustnessQuantifier, None, approximation_method="bidon"
        )

    def test_init_methods(self):
        """"""
        RobustnessQuantifier(self.database)
        RobustnessQuantifier(self.database, "BFGS")
        RobustnessQuantifier(self.database, "LEAST_SQUARES")

    def test_build_approx(self):
        """"""
        for method in RobustnessQuantifier.AVAILABLE_APPROXIMATIONS:
            rq = RobustnessQuantifier(self.database, method)
            rq.compute_approximation(funcname="rosen", last_iter=-1)

    def method_appprox(self, method, first_iter=0):
        """

        :param method: param first_iter:  (Default value = 0)
        :param first_iter:  (Default value = 0)

        """
        rq = RobustnessQuantifier(self.database, method)
        rq.compute_approximation(funcname="rosen", first_iter=first_iter, last_iter=-1)

        out = rq.compute_function_approximation(x_vars=np.ones(self.n))
        assert abs(out) < 1e-8
        x = 0.99 * np.ones(self.n)
        out = rq.compute_function_approximation(x)
        assert abs(out - rosen(x)) < 0.01
        outg = rq.compute_gradient_approximation(x)
        assert norm(outg - rosen_der(x)) / norm(rosen_der(x)) < 0.15
        x = np.ones(self.n) + (np.array(list(range(self.n))) + 1) / (10.0 + self.n)
        out = rq.compute_function_approximation(x)
        assert abs(out - rosen(x)) < 0.04
        outg = rq.compute_gradient_approximation(x)

    def test_function_error(self):
        """"""
        rq = RobustnessQuantifier(self.database)
        rq.compute_approximation(funcname="rosen", last_iter=-1)
        rq.b_mat = None
        self.assertRaises(Exception, rq.compute_function_approximation, np.ones(self.n))
        x = np.ones(self.n) + (np.array(list(range(self.n))) + 1) / (10.0 + self.n)
        self.assertRaises(Exception, rq.compute_gradient_approximation, x)

    def test_sr1_approximation_precision(self):
        """"""
        self.method_appprox("SR1")

    def test_bfgs_approximation_precision(self):
        """"""
        self.method_appprox("BFGS")

    def test_mc_average(self):
        """"""
        rq = RobustnessQuantifier(self.database)
        rq.compute_approximation(funcname="rosen")
        mu = np.ones(2)
        cov = 0.0001 * np.eye(2)
        rq.montecarlo_average_var(mu, cov)

        cov = 0.0001 * np.eye(3)
        self.assertRaises(Exception, rq.montecarlo_average_var, mu, cov)

    def test_compute_expected_value(self):
        """"""
        rq = RobustnessQuantifier(self.database)
        rq.compute_approximation(funcname="rosen")
        mu = np.ones(2)
        cov = 0.0001 * np.eye(2)
        e = rq.compute_expected_value(mu, cov)
        var = rq.compute_variance(mu, cov)
        e_ref, var_ref = rq.montecarlo_average_var(
            mu, cov, func=rosen, n_samples=300000
        )
        assert abs((e - e_ref) / e_ref) < 0.0026
        assert abs((var - var_ref) / var_ref) < 0.01

        cov = 0.0001 * np.eye(3)
        self.assertRaises(Exception, rq.compute_expected_value, mu, cov)

        cov = 0.0001 * np.eye(2)
        rq.b_mat = None
        self.assertRaises(Exception, rq.compute_expected_value, mu, cov)

    def test_compute_variance_error(self):
        """"""
        rq = RobustnessQuantifier(self.database)
        rq.compute_approximation(funcname="rosen")
        mu = np.ones(2)
        cov = 0.0001 * np.eye(3)
        self.assertRaises(Exception, rq.compute_variance, mu, cov)

        cov = 0.0001 * np.eye(2)
        rq.b_mat = None
        self.assertRaises(Exception, rq.compute_variance, mu, cov)


if __name__ == "__main__":
    unittest.main()
