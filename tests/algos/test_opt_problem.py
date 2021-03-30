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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tempfile
import timeit
import unittest
from builtins import range, str
from os.path import dirname, exists, join

import numpy as np
from future import standard_library
from numpy import array, ndarray, ones, zeros
from scipy.linalg import norm
from scipy.optimize import rosen, rosen_der

from gemseo import SOFTWARE_NAME
from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.stop_criteria import DesvarIsNan, FunctionIsNan
from gemseo.api import configure_logger
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.function import MDOFunction
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.problems.sobieski.wrappers import SobieskiProblem, SobieskiStructure
from gemseo.third_party.junitxmlreq import link_to

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)

DIRNAME = dirname(os.path.realpath(__file__))
FAIL_HDF = os.path.join(DIRNAME, "fail2.hdf5")


class Test_OptProblem(unittest.TestCase):
    """ """

    def __create_pow2_problem(self):
        design_space = DesignSpace()
        design_space.add_variable("x", 3, l_b=-1.0, u_b=1.0)
        x_0 = np.ones(3)
        design_space.set_current_x(x_0)

        problem = OptimizationProblem(design_space)
        power2 = Power2(design_space)
        problem.objective = MDOFunction(
            power2.pow2,
            name="pow2",
            f_type="obj",
            jac=power2.pow2_jac,
            expr="x[0]**2+x[1]**2+x[2]**2",
            args=["x"],
        )
        return problem

    def create_rosen_pb(self):
        """ """
        design_space = DesignSpace()
        problem = OptimizationProblem(design_space)
        problem.objective = MDOFunction(
            rosen, name="rosen", f_type="obj", jac=rosen_der
        )
        n = 3
        problem.x_0 = np.zeros(n)
        problem.u_bounds = np.ones(n)
        problem.l_bounds = -np.ones(n)
        problem.check()
        return problem

    def test_init(self):
        """ """
        design_space = DesignSpace()
        OptimizationProblem(design_space)

    def test_checks(self):
        """ """
        n = 3
        design_space = DesignSpace()
        problem = OptimizationProblem(design_space)
        problem.objective = MDOFunction(
            rosen, name="rosen", f_type="obj", jac=rosen_der
        )

        with self.assertRaises(Exception):
            problem.design_space.set_current_x(np.zeros(n))
        with self.assertRaises(Exception):
            problem.design_space.set_upper_bound("x", np.ones(n))
        with self.assertRaises(Exception):
            problem.design_space.set_lower_bound("x", -np.ones(n))

        self.assertRaises(ValueError, problem.check)
        design_space.add_variable("x")
        problem.check()

    def test_callback(self):
        """ """
        n = 3
        design_space = DesignSpace()
        design_space.add_variable("x", n, l_b=-1.0, u_b=1.0)
        design_space.set_current_x(np.zeros(n))
        problem = OptimizationProblem(design_space)
        problem.objective = MDOFunction(
            rosen, name="rosen", f_type="obj", jac=rosen_der
        )
        problem.check()

        self.i_was_called = False

        def call_me():
            """ """
            self.i_was_called = True

        problem.add_callback(call_me)
        problem.preprocess_functions()
        problem.check()

        problem.objective(problem.design_space.get_current_x())
        assert self.i_was_called

    #         design_space = DesignSpace()
    #         design_space.add_variable("x", n, l_b=-1., u_b=1.)
    #         design_space.set_current_x(np.zeros(n))
    #         problem = OptimizationProblem(design_space)
    #         problem.objective = MDOFunction(rosen, name="rosen",
    #                                         f_type="obj", jac=rosen_der)

    @link_to("Req-MDO-4.3", "Req-MDO-4.5")
    def test_add_constraints(self):
        """ """
        problem = self.__create_pow2_problem()
        ineq1 = MDOFunction(
            Power2.ineq_constraint1,
            name="ineq1",
            f_type="ineq",
            jac=Power2.ineq_constraint1_jac,
            expr="0.5 -x[0] ** 3",
            args=["x"],
        )
        problem.add_ineq_constraint(ineq1, value=-1)
        self.assertEqual(problem.get_ineq_constraints_number(), 1)
        self.assertEqual(problem.get_eq_constraints_number(), 0)

        problem.add_ineq_constraint(ineq1, value=-1)
        problem.add_ineq_constraint(ineq1, value=-1)

        self.assertEqual(problem.get_constraints_number(), 3)
        self.assertTrue(problem.has_nonlinear_constraints())

        ineq2 = MDOFunction(Power2.ineq_constraint1, name="ineq2")
        self.assertRaises(
            Exception, problem.add_constraint, ineq2, value=None, cstr_type=None
        )

        problem.add_constraint(ineq1, positive=True)

        problem = self.__create_pow2_problem()
        problem.constraints = [problem.objective]
        self.assertRaises(ValueError, problem.check)

    def test_get_dimension(self):
        """ """
        problem = self.__create_pow2_problem()
        problem.u_bounds = None
        problem.l_bounds = None
        dim = 3
        self.assertEqual(problem.get_dimension(), dim)
        problem.u_bounds = np.ones(3)
        self.assertEqual(problem.get_dimension(), dim)
        problem.l_bounds = -np.ones(3)
        self.assertEqual(problem.get_dimension(), dim)

    def test_check_format(self):
        """ """
        problem = self.__create_pow2_problem()
        self.assertRaises(TypeError, problem.check_format, "1")

    @link_to("Req-MDO-4.3", "Req-MDO-4.5")
    def test_constraints_dim(self):
        """ """
        problem = self.__create_pow2_problem()
        ineq1 = MDOFunction(
            Power2.ineq_constraint1,
            name="ineq1",
            f_type="ineq",
            jac=Power2.ineq_constraint1_jac,
            expr="0.5 -x[0] ** 3",
            args=["x"],
        )
        problem.add_ineq_constraint(ineq1, value=-1)
        self.assertRaises(Exception, problem.get_ineq_cstr_total_dim)
        self.assertEqual(problem.get_eq_constraints_number(), 0)
        assert len(problem.get_nonproc_constraints()) == 0
        problem.preprocess_functions()
        assert len(problem.get_nonproc_constraints()) == 1

    def test_check(self):
        """ """
        # Objective is missing!
        design_space = DesignSpace()
        design_space.add_variable("x", 3, l_b=-1.0, u_b=1.0)
        design_space.set_current_x(np.array([1.0, 1.0, 1.0]))
        problem = OptimizationProblem(design_space)
        self.assertRaises(Exception, problem.check)

    @link_to("Req-MDO-4.4", "Req-MDO-4.5")
    def test_missing_constjac(self):
        """ """
        problem = self.__create_pow2_problem()

        ineq1 = MDOFunction(sum, name="sum", f_type="ineq", expr="sum(x)", args=["x"])
        problem.add_ineq_constraint(ineq1, value=-1)
        problem.preprocess_functions()
        self.assertRaises(
            ValueError, problem.evaluate_functions, ones(3), eval_jac=True
        )

    def _test_check_bounds(self):
        """ """
        dim = 3
        problem = self.__create_pow2_problem()
        problem.x_0 = np.ones(dim)

        problem.design_space.set_upper_bound("x", np.ones(dim))
        problem.design_space.set_lower_bound("x", np.array(dim * [-1]))
        self.assertRaises(TypeError, problem.check)

        problem.design_space.set_lower_bound("x", np.ones(dim))
        problem.design_space.set_upper_bound("x", np.array(dim * [-1]))
        self.assertRaises(TypeError, problem.check)

        problem.design_space.set_lower_bound("x", -np.ones(dim + 1))
        problem.design_space.set_upper_bound("x", np.ones(dim))
        self.assertRaises(ValueError, problem.check)

        problem.design_space.set_lower_bound("x", -np.ones(dim))
        problem.design_space.set_upper_bound("x", np.ones(dim))
        x_0 = np.ones(dim + 1)
        problem.design_space.set_current_x(x_0)
        self.assertRaises(ValueError, problem.check)

        problem.design_space.set_lower_bound("x", np.ones(dim) * 2)
        problem.design_space.set_upper_bound("x", np.ones(dim))
        x_0 = np.ones(dim)
        problem.design_space.set_current_x(x_0)
        self.assertRaises(ValueError, problem.check)

    def test_pb_type(self):
        """ """
        problem = self.__create_pow2_problem()
        problem.pb_type = "None"
        self.assertRaises(TypeError, problem.check)

    @link_to("Req-MDO-4.4", "Req-MDO-4.5")
    def test_differentiation_method(self):
        """ """
        problem = self.__create_pow2_problem()
        problem.differentiation_method = "None"
        self.assertRaises(ValueError, problem.check)

        problem.differentiation_method = problem.COMPLEX_STEP
        problem.fd_step = 0.0
        self.assertRaises(ValueError, problem.check)
        problem.fd_step = 1e-7 + 1j * 1.0e-7
        problem.check()
        problem.fd_step = 1j * 1.0e-7
        problem.check()

        problem.differentiation_method = problem.FINITE_DIFFERENCES
        problem.fd_step = 0.0
        self.assertRaises(ValueError, problem.check)
        problem.fd_step = 1e-7 + 1j * 1.0e-7
        problem.check()

    def test_get_dv_names(self):
        """ """
        problem = Power2()
        OptimizersFactory().execute(problem, "SLSQP")
        self.assertListEqual(problem.design_space.variables_names, ["x"])

    def test_get_best_infeasible_point(self):
        problem = Power2()
        x_opt, f_opt, is_opt_feasible, _ = problem.get_best_infeasible_point()
        assert x_opt is None
        assert f_opt is None
        assert not is_opt_feasible
        problem.preprocess_functions()
        x_0 = problem.design_space.normalize_vect(zeros(3))
        f_val = problem.objective(x_0)
        x_opt, f_opt, is_opt_feasible, opt_fd = problem.get_best_infeasible_point()
        assert is_opt_feasible
        assert (x_opt == zeros(3)).all()
        assert f_opt == f_val
        assert "pow2" in opt_fd

        problem = Power2()
        problem.preprocess_functions()
        x_1 = problem.design_space.normalize_vect(array([-1.0, 0.0, 0.0]))
        problem.evaluate_functions(x_1)
        x_2 = problem.design_space.normalize_vect(array([0.0, -1.0, 0.0]))
        problem.evaluate_functions(x_2)
        x_opt, f_opt, is_opt_feasible, opt_fd = problem.get_best_infeasible_point()
        assert not is_opt_feasible
        assert x_opt is not None
        assert f_opt is not None
        assert len(opt_fd) > 0

    def test_feasible_optimum_points(self):
        """ """
        problem = Power2()
        self.assertRaises(ValueError, problem.get_optimum)
        OptimizersFactory().execute(
            problem, "SLSQP", eq_tolerance=1e-6, ineq_tolerance=1e-6
        )
        feasible_points, _ = problem.get_feasible_points()
        assert len(feasible_points) == 2
        min_value, solution, is_feasible, _, _ = problem.get_optimum()
        assert (solution == feasible_points[-1]).all()
        self.assertAlmostEqual(min_value, 2.192090802, 9)
        self.assertAlmostEqual(solution[0], 0.79370053, 8)
        self.assertAlmostEqual(solution[1], 0.79370053, 8)
        self.assertAlmostEqual(solution[2], 0.96548938, 8)
        assert is_feasible

    #         print "*", solution
    #         assert problem.is_feasible_point(solution, 1e-6, 1e-6)

    def test_nan(self):
        problem = Power2()
        problem.preprocess_functions()

        self.assertRaises(DesvarIsNan, problem.objective, array([1.0, float("nan")]))
        self.assertRaises(
            FunctionIsNan, problem.objective.jac, array([1.0, float("nan")])
        )
        problem = Power2()
        problem.objective.jac = lambda x: array([float("nan")] * 3)
        problem.preprocess_functions()
        self.assertRaises(FunctionIsNan, problem.objective.jac, array([0.1, 0.2, 0.3]))

    def test_preprocess_functons(self):
        problem = Power2()
        problem.preprocess_functions(normalize=False, round_ints=False)

    @link_to("Req-MDO-4.4", "Req-MDO-4.5", "Req-MDO-4.6")
    def test_export_hdf(self):
        """ """
        file_path = join(tempfile.mkdtemp(), "power2.hdf5")
        file_path = "power2.hdf5"
        problem = Power2()
        OptimizersFactory().execute(problem, "SLSQP")
        problem.export_hdf(file_path, append=True)  # Shall still work now

        def check_pb(imp_pb):
            assert exists(file_path)
            repr_pb = str(problem)
            repr_sol = str(problem.solution)
            imp_pb = OptimizationProblem.import_hdf(file_path)
            assert str(imp_pb) == repr_pb
            assert str(imp_pb.solution) == repr_sol
            assert exists(file_path)

            assert problem.get_eq_cstr_total_dim() == 1
            assert problem.get_ineq_cstr_total_dim() == 2

        problem.export_hdf(file_path, append=False)

        imp_pb = OptimizationProblem.import_hdf(file_path)
        check_pb(imp_pb)

        problem.export_hdf(file_path, append=True)
        imp_pb = OptimizationProblem.import_hdf(file_path)
        check_pb(imp_pb)
        # remove(file_path)
        val = imp_pb.objective(imp_pb.database.get_x_by_iter(1))
        assert isinstance(val, float)
        jac = imp_pb.objective.jac(imp_pb.database.get_x_by_iter(0))
        assert isinstance(jac, ndarray)
        self.assertRaises(ValueError, imp_pb.objective, array([1.1254]))

    def test_evaluate_functions(self):
        problem = Power2()
        problem.evaluate_functions(
            x_vect=array([1.0, 0.5, 0.2]),
            eval_jac=True,
            eval_obj=False,
            normalize=False,
        )
        self.assertRaises(
            ValueError,
            problem.evaluate_functions,
            normalize=True,
            no_db_no_norm=True,
            eval_obj=False,
        )
        problem.evaluate_functions(normalize=False, no_db_no_norm=True, eval_obj=False)

    def test_no_normalization(self):
        problem = Power2()
        OptimizersFactory().execute(problem, "SLSQP", normalize_design_space=False)
        f_opt, _, is_feas, _, _ = problem.get_optimum()
        assert is_feas
        assert abs(f_opt - 2.192) < 0.01

    def test_nan_func(self):
        problem = Power2()

        def nan_func(x):
            return float("nan")

        problem.objective.func = nan_func
        problem.preprocess_functions()
        self.assertRaises(FunctionIsNan, problem.objective, zeros(3))

    def test_fail_import(self):
        self.assertRaises(KeyError, OptimizationProblem.import_hdf, FAIL_HDF)

    def test_add_listeners(self):
        problem = Power2()
        self.assertRaises(TypeError, problem.add_store_listener, "toto")
        self.assertRaises(TypeError, problem.add_new_iter_listener, "toto")

    def test_append_export(self):
        problem = Rosenbrock()
        problem.preprocess_functions()
        func = problem.objective
        file_path_db = join(tempfile.mkdtemp(), "test_pb_append.hdf5")
        # Export empty file
        problem.export_hdf(file_path_db, append=False)

        n_calls = 200
        for i in range(n_calls):
            func(array([0.1, 1.0 / (i + 1.0)]))

        # Export again with append mode
        t0 = timeit.default_timer()
        problem.export_hdf(file_path_db, append=True)
        dt1 = timeit.default_timer() - t0

        read_db = Database(file_path_db)
        assert len(read_db) == n_calls

        i += 1
        func(array([0.1, 1.0 / (i + 1.0)]))

        # Export again with append mode and check that it is much faster
        t0 = timeit.default_timer()
        problem.export_hdf(file_path_db, append=True)
        dt2 = timeit.default_timer() - t0
        read_db = Database(file_path_db)
        assert len(read_db) == n_calls + 1
        os.remove(file_path_db)

        assert dt1 / dt2 > 2.0  # 70 in practice

    def test_grad_normalization(self):
        problem = self.__create_pow2_problem()
        x_vec = ones(3)
        grad = problem.objective.jac(x_vec)
        problem.preprocess_functions(normalize=True)
        norm_grad = problem.objective.jac(x_vec)

        self.assertAlmostEqual(norm(norm_grad - 2 * grad), 0.0)

        unnorm_grad = problem.design_space.normalize_vect(norm_grad, minus_lb=False)
        self.assertAlmostEqual(norm(unnorm_grad - grad), 0.0)

    def test_2d_objective(self):
        disc = SobieskiStructure()
        design_space = SobieskiProblem().read_design_space()
        inputs = disc.get_input_data_names()
        design_space.filter(inputs)
        doe_scenario = DOEScenario([disc], "DisciplinaryOpt", "y_12", design_space)
        doe_scenario.execute({"algo": "DiagonalDOE", "n_samples": 10})

    def test_observable(self):
        problem = self.__create_pow2_problem()
        observable = MDOFunction(norm, "design norm")
        problem.add_observable(observable)

        # Check that the observable can be found
        assert problem.get_observable("design norm") is observable
        self.assertRaises(ValueError, problem.get_observable, "toto")

        # Check that the observable is stored in the database
        OptimizersFactory().execute(problem, "SLSQP")
        database = problem.database
        iter_norms = [norm(key.unwrap()) for key in database.keys()]
        iter_obs = [value["design norm"] for value in database.values()]
        assert iter_obs == iter_norms

        # Check that the observable is exported
        dataset = problem.export_to_dataset("dataset")
        func_data = dataset.get_data_by_group("functions", as_dict=True)
        obs_data = func_data.get("design norm")
        assert obs_data is not None
        assert func_data["design norm"][:, 0].tolist() == iter_norms
