# -*- coding: utf-8 -*-
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

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from builtins import next, str
from os.path import dirname, join

import numpy as np
from future import standard_library
from numpy import array, multiply, outer
from numpy.linalg import LinAlgError, norm
from scipy.optimize import rosen_hess

from gemseo import SOFTWARE_NAME
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import configure_logger
from gemseo.post.core.hessians import (
    BFGSApprox,
    HessianApproximation,
    LSTSQApprox,
    SR1Approx,
)
from gemseo.problems.analytical.rosenbrock import Rosenbrock

standard_library.install_aliases()


LOGGER = configure_logger(SOFTWARE_NAME)

MDF_HIST = join(dirname(__file__), "mdf_history.h5")


class Test_BFGSApprox(unittest.TestCase):
    """ """

    def build_history(self, n, l_b=-2, u_b=2.0):
        """

        :param n:

        """
        self.n = n
        self.problem = Rosenbrock(n, l_b=l_b, u_b=u_b)
        self.database = self.problem.database
        result = OptimizersFactory().execute(self.problem, "SLSQP", max_iter=400)
        self.x_opt = result.x_opt
        self.assertAlmostEqual(np.linalg.norm(self.x_opt - np.ones(n)), 0.0, 2)
        self.H_ref = rosen_hess(self.x_opt)
        LOGGER.info("niter = " + str(result.n_obj_call))

    def compare_approximations(self, approx_class, ermax=0.7, **kwargs):
        """

        :param approx_class: param ermax:  (Default value = 0.7)
        :param ermax:  (Default value = 0.7)
        :param **kwargs:

        """
        approx = approx_class(self.database)
        H_approx, _, _, _ = approx.build_approximation(
            funcname=self.problem.objective.name, **kwargs
        )
        assert H_approx.shape == (self.n, self.n)
        error = self.compute_error(H_approx, approx_class)
        if error > ermax:
            LOGGER.error("Exact  hessian  : \n" + str(self.H_ref))
            LOGGER.error("Approx hessian  : \n" + str(H_approx))
            raise Exception("Approximation failed")

    def test_scaling(self):
        self.build_history(2, l_b=array([-2.0, -1.0]), u_b=array([2.0, 1.0]))
        approx = HessianApproximation(self.database)
        design_space = self.problem.design_space
        H_approx_unscaled, _, _, _ = approx.build_approximation(
            funcname=self.problem.objective.name,
            scaling=True,
            design_space=design_space,
            normalize_design_space=False,
        )

        approx = SR1Approx(self.database)
        design_space = self.problem.design_space
        H_approx_unscaled, _, _, _ = approx.build_approximation(
            funcname=self.problem.objective.name,
            scaling=True,
            design_space=design_space,
            normalize_design_space=False,
        )

        H_approx_scaled, _, _, _ = approx.build_approximation(
            funcname=self.problem.objective.name,
            scaling=True,
            design_space=design_space,
            normalize_design_space=True,
        )

        H_exact = rosen_hess(self.x_opt)

        v = design_space._norm_factor
        scale_fact = outer(v, v.T)

        H_exact_scaled = multiply(H_exact, scale_fact)
        H_approx_unscaled_scaled = multiply(H_approx_unscaled, scale_fact)
        assert (
            norm(H_exact_scaled - H_approx_unscaled_scaled) / norm(H_exact_scaled)
            < 1e-2
        )
        assert norm(H_exact_scaled - H_approx_scaled) / norm(H_exact_scaled) < 1e-2

    def compute_error(self, H_approx, approx_class):
        """

        :param H_approx: param approx_class:
        :param approx_class:

        """
        error = (norm(H_approx - self.H_ref) / norm(self.H_ref)) * 100
        LOGGER.info(approx_class.__name__ + " H Error = " + str(error) + " %")
        return error

    def test_baseclass_methods(self):
        """ """
        self.build_history(2)
        apprx = HessianApproximation(self.database)
        # 73 items in database
        at_most_niter = 2
        x_hist, x_grad_hist, n_iter, _ = apprx.get_x_grad_history(
            self.problem.objective.name, at_most_niter=at_most_niter
        )
        assert n_iter == at_most_niter
        assert x_hist.shape[0] == at_most_niter
        assert x_grad_hist.shape[0] == at_most_niter

        _, _, n_iter_ref, nparam = apprx.get_x_grad_history(
            self.problem.objective.name, at_most_niter=-1
        )

        _, _, n_iter_2, _ = apprx.get_x_grad_history(
            self.problem.objective.name, last_iter=n_iter_ref
        )

        assert n_iter_ref == n_iter_2
        _, _, n_iter_3, _ = apprx.get_x_grad_history(
            self.problem.objective.name, first_iter=10
        )

        assert n_iter_ref == n_iter_3 + 10

        apprx.build_approximation(
            self.problem.objective.name, b_mat0=np.eye(nparam), save_matrix=True
        )

        assert len(apprx.b_mat_history) > 1

        self.assertRaises(
            ValueError,
            apprx.get_x_grad_history,
            self.problem.objective.name,
            at_most_niter=1,
        )
        self.database.clear()

        self.assertRaises(
            ValueError,
            apprx.get_x_grad_history,
            self.problem.objective.name,
            at_most_niter=at_most_niter,
        )

        self.assertRaises(
            ValueError,
            apprx.get_x_grad_history,
            self.problem.objective.name,
            at_most_niter=at_most_niter,
            normalize_design_space=True,
        )

    def test_get_x_grad_history_on_sobieski(self):
        opt_pb = OptimizationProblem.import_hdf(MDF_HIST)
        apprx = HessianApproximation(opt_pb.database)
        self.assertRaises(ValueError, apprx.get_x_grad_history, "g_1")
        x_hist, x_grad_hist, n_iter, nparam = apprx.get_x_grad_history(
            "g_1", func_index=1
        )

        assert len(x_hist) == 4
        assert n_iter == 4
        assert nparam == 10
        for x in x_hist:
            assert x.shape == (nparam,)

        assert len(x_hist) == len(x_grad_hist)
        for grad in x_grad_hist:
            assert grad.shape == (nparam,)

        self.assertRaises(ValueError, apprx.get_s_k_y_k, x_hist, x_grad_hist, 5)

        self.assertRaises(ValueError, apprx.get_x_grad_history, "g_1", func_index=7)

        # Create inconsistent optimization history by restricting g_2 gradient
        # size
        x_0 = next(iter(opt_pb.database.keys()))
        val_0 = opt_pb.database[x_0]
        val_0["@g_2"] = val_0["@g_2"][1:]
        self.assertRaises(ValueError, apprx.get_x_grad_history, "g_2")

    def test_n_2(self):
        """ """
        self.build_history(2)
        self.compare_approximations(BFGSApprox, first_iter=8, ermax=3.0)
        self.compare_approximations(LSTSQApprox, first_iter=13, ermax=20.0)
        self.compare_approximations(SR1Approx, first_iter=7, ermax=30.0)
        self.compare_approximations(HessianApproximation, first_iter=8, ermax=30.0)

    def test_n_5(self):
        """ """
        self.build_history(5)
        self.compare_approximations(BFGSApprox, first_iter=5, ermax=30.0)
        self.compare_approximations(LSTSQApprox, first_iter=19, ermax=40.0)
        self.compare_approximations(SR1Approx, first_iter=5, ermax=30.0)
        self.compare_approximations(HessianApproximation, first_iter=5, ermax=30.0)

    def test_n_35(self):
        """ """
        self.build_history(35)
        self.compare_approximations(SR1Approx, first_iter=60, ermax=40.0)
        self.compare_approximations(LSTSQApprox, first_iter=165, ermax=110.0)
        self.compare_approximations(SR1Approx, first_iter=60, ermax=40.0)
        self.compare_approximations(HessianApproximation, first_iter=60, ermax=40.0)

    def test_build_inverse_approximation(self):
        self.build_history(2, l_b=array([-2.0, -1.0]), u_b=array([2.0, 1.0]))
        approx = HessianApproximation(self.database)
        funcname = self.problem.objective.name
        approx.build_inverse_approximation(funcname=funcname, h_mat0=[], factorize=True)
        self.assertRaises(
            LinAlgError,
            approx.build_inverse_approximation,
            funcname=funcname,
            h_mat0=array([1.0, 2.0]),
        )
        self.assertRaises(
            LinAlgError,
            approx.build_inverse_approximation,
            funcname=funcname,
            h_mat0=array([[0.0, 1.0], [-1.0, 0.0]]),
            factorize=True,
        )
        approx.build_inverse_approximation(
            funcname=funcname, h_mat0=array([[1.0, 0.0], [0.0, 1.0]]), factorize=True
        )
        approx.build_inverse_approximation(funcname=funcname, return_x_grad=True)
        x_hist, x_grad_hist, _, _ = approx.get_x_grad_history(
            self.problem.objective.name
        )
        x_corr, grad_corr = approx.compute_corrections(x_hist, x_grad_hist)
        approx.rebuild_history(x_corr, x_hist[0], grad_corr, x_grad_hist[0])
