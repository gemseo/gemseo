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
#      :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from copy import copy
from math import sqrt
from unittest import TestCase

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.lib_pdfo import PDFOOpt
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt.opt_lib import OptimizationLibrary as OptLib
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import execute_algo
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import isnan
from numpy import nan
from scipy.optimize import rosen
from scipy.optimize import rosen_der

from .opt_lib_test_base import OptLibraryTestBase

pytest.importorskip("pdfo", reason="pdfo is not available")


class TestPDFO(TestCase):
    OPT_LIB_NAME = "PDFOOpt"

    def test_init(self):
        """Tests the initialization of the problem."""
        factory = OptimizersFactory()
        if factory.is_available(self.OPT_LIB_NAME):
            factory.create(self.OPT_LIB_NAME)

    def test_failed(self):
        """"""
        algo_name = "PDFO_COBYLA"
        self.assertRaises(
            Exception,
            OptLibraryTestBase.generate_error_test,
            "PDFOAlgorithms",
            algo_name=algo_name,
            max_iter=10,
        )

    def test_nan_handling(self):
        """Test that an occurence of NaN value in the objective function does not stop
        the optimizer.

        In this case, a NaN "bubble" is put at the beginning of the optimizer path. In
        this test, it is expected that the optimizer will encouter and by-pass the NaN
        bubble.
        """
        opt_problem = Rosenbrock()

        fun = copy(opt_problem.objective)

        def wrapped_fun(x_vec):
            x = x_vec[0]
            y = x_vec[1]

            d = sqrt((x - 0.1) ** 2 + (y - 0.1) ** 2)

            if d < 0.05:
                return nan
            else:
                return fun(x_vec)

        opt_problem.objective._func = wrapped_fun
        opt_problem.stop_if_nan = False

        algo_options = {"max_iter": 10000, "rhobeg": 0.1, "rhoend": 1e-6}

        opt_result = execute_algo(
            opt_problem, "PDFO_COBYLA", algo_type="opt", **algo_options
        )

        obj_history = opt_problem.database.get_func_history("rosen")

        is_nan = any(isnan(obj_history))
        assert is_nan
        assert opt_result.f_opt < 1e-3

    def test_nan_handling_2(self):
        """Test that an occurence of NaN value in the objective function does not stop
        the optimizer.

        In this test, all the values of x>0.7 are not realizable. The optimum is then
        expected for x[0] ~= 0.7
        """
        opt_problem = Rosenbrock()

        fun = copy(opt_problem.objective)

        def wrapped_fun(x_vec):
            x = x_vec[0]

            if 0.7 < x:
                return nan
            else:
                return fun(x_vec)

        opt_problem.objective._func = wrapped_fun
        opt_problem.stop_if_nan = False

        algo_options = {"max_iter": 10000, "rhobeg": 0.1, "rhoend": 1e-6}

        opt_result = execute_algo(
            opt_problem, "PDFO_COBYLA", algo_type="opt", **algo_options
        )

        obj_history = opt_problem.database.get_func_history("rosen")

        is_nan = any(isnan(obj_history))
        assert is_nan
        assert pytest.approx(opt_result.x_opt[0], rel=1e-3) == 0.7

    def test_xtol_ftol_activation(self):
        def run_pb(algo_options):
            design_space = DesignSpace()
            design_space.add_variable("x1", 2, DesignSpace.FLOAT, -1.0, 1.0, 0.0)
            problem = OptimizationProblem(design_space)
            problem.objective = MDOFunction(rosen, "Rosenbrock", "obj", rosen_der)
            res = OptimizersFactory().execute(problem, "PDFO_COBYLA", **algo_options)
            return res, problem

        for tol_name in (
            OptLib.F_TOL_ABS,
            OptLib.F_TOL_REL,
            OptLib.X_TOL_ABS,
            OptLib.X_TOL_REL,
        ):
            res, pb = run_pb({tol_name: 1e10})
            assert tol_name in res.message
            # Check that the criteria is activated as ap
            assert len(pb.database) == 3


def get_options(algo_name):
    """
    :param algo_name:
    """
    return {"max_iter": 10000, "rhobeg": 0.3, "rhoend": 1e-6}


def get_pb_options(pb_name):
    """
    :param algo_name:
    """
    if pb_name == "Power2":
        return {"initial_value": 0.0}
    else:
        return {}


suite_tests = OptLibraryTestBase()
for test_method in suite_tests.generate_test("PDFOOpt", get_options, get_pb_options):
    setattr(TestPDFO, test_method.__name__, test_method)


def test_library_name():
    """Check the library name."""
    assert PDFOOpt.LIBRARY_NAME == "PDFO"
