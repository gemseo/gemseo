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

from __future__ import division, unicode_literals

import unittest
from copy import deepcopy
from os.path import dirname, join

import numpy as np
from numpy import array

from gemseo.algos.lagrange_multipliers import LagrangeMultipliers
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.api import create_discipline, create_scenario
from gemseo.problems.analytical.power_2 import Power2
from gemseo.utils.derivatives_approx import comp_best_step

DS_FILE = join(dirname(__file__), "sobieski_design_space.txt")


class TestLagrangeMultipliers(unittest.TestCase):
    """"""

    NLOPT_OPTIONS = {
        "eq_tolerance": 1e-11,
        "ftol_abs": 1e-14,
        "ftol_rel": 1e-14,
        "ineq_tolerance": 1e-11,
        "normalize_design_space": False,
        "xtol_abs": 1e-14,
        "xtol_rel": 1e-14,
    }

    def test_lagrange_notanoptproblem(self):
        self.assertRaises(ValueError, LagrangeMultipliers, "not_a_problem")

    def test_lagrange_solutionisnone(self):
        problem = Power2()
        self.assertRaises(ValueError, LagrangeMultipliers, problem)

    def test_lagrange_pow2_too_many_acts(self):
        """"""
        problem = Power2()
        problem.design_space.set_current_x(array([0.5, 0.9, -0.5]))
        problem.design_space.set_lower_bound("x", array([-1.0, 0.8, -1.0]))
        problem.design_space.set_upper_bound("x", array([1.0, 1.0, 0.9]))
        OptimizersFactory().execute(
            problem, "SLSQP", eq_tolerance=1e-6, ineq_tolerance=1e-6
        )
        lagrange = LagrangeMultipliers(problem)
        x_opt = problem.solution.x_opt
        x_n = problem.design_space.normalize_vect(x_opt)
        problem.evaluate_functions(x_n, eval_jac=True, normalize=True)
        lagrangian = lagrange.compute(x_opt)
        assert "upper_bounds" in lagrangian
        assert "lower_bounds" in lagrangian
        assert "equality" in lagrangian
        assert "inequality" not in lagrangian

    def test_lagrange_pow2_nact_ndim(self):
        """"""
        problem = Power2()
        problem.design_space.set_lower_bound("x", array([-1.0, 0.8, -1.0]))
        OptimizersFactory().execute(
            problem, "SLSQP", eq_tolerance=1e-6, ineq_tolerance=1e-6
        )
        lagrange = LagrangeMultipliers(problem)
        x_opt = problem.solution.x_opt
        x_n = problem.design_space.normalize_vect(x_opt)
        problem.evaluate_functions(x_n, eval_jac=True, normalize=True)
        lagrangian = lagrange.compute(x_opt)
        assert "upper_bounds" not in lagrangian
        assert "lower_bounds" in lagrangian
        assert "equality" in lagrangian
        assert "inequality" in lagrangian

    def test_lagrangian_validation_lbound(self):
        problem = Power2()
        problem.design_space.set_lower_bound("x", array([-1.0, 0.8, -1.0]))
        OptimizersFactory().execute(problem, "NLOPT_SLSQP", **self.NLOPT_OPTIONS)
        lagrange = LagrangeMultipliers(problem)
        lagrangian = lagrange.compute(problem.solution.x_opt)

        def obj(lb):
            problem = Power2()
            dspace = problem.design_space
            dspace.set_current_x(array([1.0, 0.9, 1.0]))
            dspace.set_lower_bound("x", array([-1.0, 0.8 + lb, -1.0]))
            OptimizersFactory().execute(problem, "NLOPT_SLSQP", **self.NLOPT_OPTIONS)
            return problem.solution.f_opt

        eps = 1e-5
        df_fd = (obj(eps) - obj(-eps)) / (2 * eps)
        df_anal = lagrangian["lower_bounds"][1]
        err = abs((df_fd - df_anal) / df_anal)
        assert err < 1e-7

    def test_lagrangian_validation_lbound_normalize(self):
        problem = Power2()
        options = deepcopy(self.NLOPT_OPTIONS)
        options["normalize_design_space"] = True
        problem.design_space.set_lower_bound("x", array([-1.0, 0.8, -1.0]))
        OptimizersFactory().execute(problem, "NLOPT_SLSQP", **options)
        lagrange = LagrangeMultipliers(problem)
        lagrangian = lagrange.compute(problem.solution.x_opt)

        def obj(lb):
            problem = Power2()
            dspace = problem.design_space
            dspace.set_current_x(array([1.0, 0.9, 1.0]))
            dspace.set_lower_bound("x", array([-1.0, 0.8 + lb, -1.0]))
            OptimizersFactory().execute(problem, "NLOPT_SLSQP", **options)
            return problem.solution.f_opt

        eps = 1e-3
        df_fd = (obj(eps) - obj(-eps)) / (2 * eps)
        df_anal = lagrangian["lower_bounds"][1]
        err = abs((df_fd - df_anal) / df_anal)
        assert err < 1e-8

    def test_lagrangian_validation_eq(self):
        problem = Power2()
        OptimizersFactory().execute(problem, "NLOPT_SLSQP", **self.NLOPT_OPTIONS)

        lagrange = LagrangeMultipliers(problem)
        lagrangian = lagrange.compute(problem.solution.x_opt)

        def obj(eq_val):
            problem2 = Power2()
            problem2.constraints[-1] = problem2.constraints[-1] + eq_val
            OptimizersFactory().execute(problem2, "NLOPT_SLSQP", **self.NLOPT_OPTIONS)
            return problem2.solution.f_opt

        eps = 1e-5
        df_fd = (obj(eps) - obj(-eps)) / (2 * eps)
        df_anal = lagrangian["equality"][1]
        err = abs((df_fd - df_anal) / df_fd)
        assert err < 1e-7

    def test_lagrangian_validation_ineq_normalize(self):

        options = deepcopy(self.NLOPT_OPTIONS)
        options["normalize_design_space"] = True

        def obj(eq_val):
            problem2 = Power2()
            problem2.constraints[-2] = problem2.constraints[-2] + eq_val
            OptimizersFactory().execute(problem2, "NLOPT_SLSQP", **options)
            return problem2.solution.f_opt

        def obj_grad(eq_val):
            problem = Power2()
            problem.constraints[-2] = problem.constraints[-2] + eq_val
            OptimizersFactory().execute(problem, "NLOPT_SLSQP", **options)
            lagrange = LagrangeMultipliers(problem)
            x_opt = problem.solution.x_opt
            lagrangian = lagrange.compute(x_opt)
            df_anal = lagrangian["inequality"][1][1]

            return df_anal

        eps = 1e-4
        obj_ref = obj(0.0)

        _, _, opt_step = comp_best_step(obj(eps), obj_ref, obj(-eps), eps, 1e-8)
        df_anal = obj_grad(0.0)

        df_fd = (obj(opt_step) - obj(-opt_step)) / (2 * opt_step)
        err = abs((df_fd - df_anal) / df_fd)
        assert err < 1e-3

    def test_lagrangian_eq(self):
        disciplines = create_discipline(
            [
                "SobieskiStructure",
                "SobieskiPropulsion",
                "SobieskiAerodynamics",
                "SobieskiMission",
            ]
        )
        scenario = create_scenario(
            disciplines,
            formulation="MDF",
            objective_name="y_4",
            design_space=DS_FILE,
            tolerance=1e-12,
            max_mda_iter=20,
            warm_start=True,
            maximize_objective=True,
            use_lu_fact=True,
            linear_solver_tolerance=1e-15,
        )
        for cstr in ["g_1", "g_2", "g_3"]:
            scenario.add_constraint(cstr, "eq")
        run_inputs = {
            "max_iter": 10,
            "algo": "SLSQP",
            "algo_options": {
                "ftol_rel": 1e-10,
                "ineq_tolerance": 2e-3,
                "normalize_design_space": True,
            },
        }
        scenario.execute(run_inputs)
        problem = scenario.formulation.opt_problem
        lagrange = LagrangeMultipliers(problem)
        lagrange.compute(problem.solution.x_opt)

    def test_lagrangian_ineq(self):
        disciplines = create_discipline(
            [
                "SobieskiStructure",
                "SobieskiPropulsion",
                "SobieskiAerodynamics",
                "SobieskiMission",
            ]
        )
        scenario = create_scenario(
            disciplines,
            formulation="MDF",
            objective_name="y_4",
            design_space=DS_FILE,
            tolerance=1e-12,
            max_mda_iter=20,
            warm_start=True,
            maximize_objective=True,
            use_lu_fact=True,
            linear_solver_tolerance=1e-15,
        )
        for cstr in ["g_1", "g_2", "g_3"]:
            scenario.add_constraint(cstr, "ineq")
        run_inputs = {
            "max_iter": 10,
            "algo": "SLSQP",
            "algo_options": {
                "ftol_rel": 1e-10,
                "ineq_tolerance": 2e-3,
                "normalize_design_space": True,
            },
        }
        scenario.execute(run_inputs)
        problem = scenario.formulation.opt_problem
        lagrange = LagrangeMultipliers(problem)
        lagrange.compute(problem.solution.x_opt)

    def test_lagrange_store(self):
        problem = Power2()
        options = deepcopy(self.NLOPT_OPTIONS)
        options["normalize_design_space"] = True
        OptimizersFactory().execute(problem, "NLOPT_SLSQP", **options)
        lagrange = LagrangeMultipliers(problem)
        lagrange.active_lb_names = [0]
        lagrange._store_multipliers(np.ones(10))
        lagrange.active_lb_names = []
        lagrange.active_ub_names = [0]
        lagrange._store_multipliers(-1 * np.ones(10))
        lagrange.active_lb_names = []
        lagrange.active_ub_names = []
        lagrange.active_ineq_names = [0]
        lagrange._store_multipliers(-1 * np.ones(10))
