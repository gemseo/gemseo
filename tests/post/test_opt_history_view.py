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
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import division, unicode_literals

import unittest
from os.path import dirname, exists, join

import pytest

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import execute_post
from gemseo.post.opt_history_view import OptHistoryView
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rosenbrock import Rosenbrock

DIRNAME = dirname(__file__)
POWER2 = join(DIRNAME, "power2_opt_pb.h5")
POWER2_NAN = join(DIRNAME, "power2_opt_pb_nan.h5")


@pytest.mark.usefixtures("tmp_wd")
class TestPlotHistoryViews(unittest.TestCase):
    """"""

    def test_view(self):
        """"""
        problem = Rosenbrock()
        OptimizersFactory().execute(problem, "L-BFGS-B")
        view = OptHistoryView(problem)
        path = "rosen_1"
        view.execute(show=False, save=True, file_path=path)
        for full_path in view.output_files:
            assert exists(full_path)

    def test_view_load_pb(self):
        """"""
        problem = OptimizationProblem.import_hdf(POWER2)
        view = OptHistoryView(problem)
        view.execute(show=False, save=True, file_path="power2view", obj_relative=True)
        for full_path in view.output_files:
            assert exists(full_path)

    def test_view_constraints(self):
        """"""
        problem = Power2()
        OptimizersFactory().execute(problem, "SLSQP")
        view = OptHistoryView(problem)

        _, cstr = view._get_constraints(["toto", "ineq1"])
        assert len(cstr) == 1
        view.execute(
            show=False,
            save=True,
            variables_names=["x"],
            file_path="power2_2",
            obj_min=0.0,
            obj_max=5.0,
        )
        for full_path in view.output_files:
            assert exists(full_path)

    def test_nans(self):

        #         problem = Power2()
        #         refun = problem.objective._func
        #         refctr = problem.constraints[0]._func
        #
        #         def newpt(x):
        #             out = refun(x)
        #             if x[1] < 0.51:
        #                 out = float("nan")
        #             return out
        #
        #         def newptc(x):
        #             out = refctr(x)
        #             if x[1] < 0.51:
        #                 out = float("nan")
        #             return out
        #
        #         problem.objective._func = newpt
        #         problem.objective.name = "Pow2 with nans"
        #         problem.constraints[0]._func = newptc
        #         problem.stop_if_nan = False
        #
        #         execute_algo(problem, "NLOPT_COBYLA", max_iter=10)
        #
        #         problem.export_hdf("power2_opt_pb_nan.h5")

        problem = OptimizationProblem.import_hdf(POWER2_NAN)
        view = execute_post(problem, "OptHistoryView", show=False, save=True)

        for full_path in view.output_files:
            assert exists(full_path)
