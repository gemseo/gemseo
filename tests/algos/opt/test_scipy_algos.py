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
#      :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

from unittest import TestCase

from future import standard_library
from numpy import inf
from scipy.optimize.optimize import rosen, rosen_der

from gemseo import SOFTWARE_NAME
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import configure_logger
from gemseo.core.function import MDOFunction
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.third_party.junitxmlreq import link_to

from .opt_lib_test_base import OptLibraryTestBase

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class TestScipy(TestCase):
    """ """

    OPT_LIB_NAME = "ScipyOpt"

    @link_to("Req-DEP-1")
    def test_init(self):
        """ """
        factory = OptimizersFactory()
        if factory.is_available(self.OPT_LIB_NAME):
            factory.create(self.OPT_LIB_NAME)

    def test_display(self):
        """ """
        algo_name = "SLSQP"
        OptLibraryTestBase.generate_one_test(
            self.OPT_LIB_NAME, algo_name=algo_name, max_iter=10, disp=10
        )

    def test_handles_cstr(self):
        """ """
        algo_name = "TNC"
        self.assertRaises(
            Exception,
            OptLibraryTestBase.generate_one_test,
            self.OPT_LIB_NAME,
            algo_name=algo_name,
            max_iter=10,
        )

    def test_algorithm_suited(self):
        """ """
        algo_name = "SLSQP"
        opt_library = OptLibraryTestBase.generate_one_test(
            self.OPT_LIB_NAME, algo_name=algo_name, max_iter=10
        )

        self.assertFalse(
            opt_library.is_algorithm_suited(
                opt_library.lib_dict["TNC"], opt_library.problem
            )
        )

        opt_library.problem.pb_type = "non-linear"
        opt_library.lib_dict["SLSQP"]["pb_type"] = "linear"
        self.assertFalse(
            opt_library.is_algorithm_suited(
                opt_library.lib_dict["SLSQP"], opt_library.problem
            )
        )

    def test_positive_constraints(self):
        """ """
        algo_name = "SLSQP"
        opt_library = OptLibraryTestBase.generate_one_test(
            self.OPT_LIB_NAME, algo_name=algo_name, max_iter=10
        )
        self.assertTrue(opt_library.is_algo_requires_positive_cstr(algo_name))
        self.assertFalse(opt_library.is_algo_requires_positive_cstr("TNC"))

    def test_fail_opt(self):
        """ """
        algo_name = "SLSQP"
        problem = Rosenbrock()

        def i_fail(x):
            if rosen(x) < 1e-3:
                raise Exception(x)
            return rosen(x)

        problem.objective = MDOFunction(i_fail, "rosen")
        self.assertRaises(Exception, OptimizersFactory().execute, problem, algo_name)

    @link_to("Req-MDO-4.2")
    def test_tnc_options(self):
        """ """
        algo_name = "TNC"
        OptLibraryTestBase.generate_one_test_unconstrained(
            self.OPT_LIB_NAME,
            algo_name=algo_name,
            max_iter=100,
            disp=1,
            maxCGit=178,
            pg_tol=1e-8,
            eta=-1.0,
            ftol_rel=1e-10,
            xtol_rel=1e-10,
            max_ls_step_size=0.5,
            minfev=4,
        )

    def test_lbfgsb_options(self):
        """ """
        algo_name = "L-BFGS-B"
        OptLibraryTestBase.generate_one_test_unconstrained(
            self.OPT_LIB_NAME,
            algo_name=algo_name,
            max_iter=100,
            disp=1,
            maxcor=12,
            pg_tol=1e-8,
            max_fun_eval=20,
        )
        self.assertRaises(
            Exception,
            OptLibraryTestBase.generate_one_test_unconstrained,
            self.OPT_LIB_NAME,
            algo_name=algo_name,
            max_iter="100",
            disp=1,
            maxcor=12,
            pg_tol=1e-8,
            max_fun_eval=1000,
        )

    def test_slsqp_options(self):
        """ """
        algo_name = "SLSQP"
        OptLibraryTestBase.generate_one_test(
            self.OPT_LIB_NAME, algo_name=algo_name, max_iter=100, disp=1, ftol_rel=1e-10
        )

    def test_normalization(self):
        """Runs a problem with one variable to be normalized
        and three not to be normalized."""
        design_space = DesignSpace()
        design_space.add_variable("x1", 1, DesignSpace.FLOAT, -1.0, 1.0, 0.0)
        design_space.add_variable("x2", 1, DesignSpace.FLOAT, -inf, 1.0, 0.0)
        design_space.add_variable("x3", 1, DesignSpace.FLOAT, -1.0, inf, 0.0)
        design_space.add_variable("x4", 1, DesignSpace.FLOAT, -inf, inf, 0.0)
        problem = OptimizationProblem(design_space)
        problem.objective = MDOFunction(rosen, "Rosenbrock", "obj", rosen_der)
        OptimizersFactory().execute(problem, "L-BFGS-B", normalize_design_space=True)
        OptimizersFactory().execute(problem, "L-BFGS-B", normalize_design_space=False)


suite_tests = OptLibraryTestBase()
for test_method in suite_tests.generate_test("SCIPY"):
    setattr(TestScipy, test_method.__name__, test_method)
