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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from future import standard_library
from numpy import array, ones, zeros
from scipy.optimize import rosen, rosen_der

from gemseo import SOFTWARE_NAME
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import (
    configure_logger,
    create_design_space,
    create_mda,
    create_scenario,
    execute_algo,
)
from gemseo.core.auto_py_discipline import AutoPyDiscipline
from gemseo.core.function import MDOFunction

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


def create_ds(N):
    design_space = create_design_space()
    design_space.add_variable("x", N, l_b=-2 * ones(N), u_b=2 * ones(N), value=zeros(N))
    return design_space


def f1(y2=1.0, z=2.0):
    y1 = z + y2
    return y1


def f2(y1=2.0, z=2.0):
    y2 = z + 2 * y1
    y3 = 14
    return y2, y3


def f3(x=1.0):
    if x > 0:
        y = -x
        return y
    y = 2 * x
    return y


def f4(x=1.0):
    if x > 0:
        y = -x
        return y
    y = 2 * x
    return y, x


class TestAutoPyDiscipline(unittest.TestCase):
    def test_basic(self):
        d1 = AutoPyDiscipline(f1)
        d1.execute()

        assert d1.local_data["y1"] == f1()

        d2 = AutoPyDiscipline(f2)
        d2.execute()
        assert d2.local_data["y2"] == f2()[0]

    def test_use_arrays(self):
        d1 = AutoPyDiscipline(f1, use_arrays=True)
        d1.execute()
        assert d1.local_data["y1"] == f1()
        d1.execute({"x1": array([1.0]), "z": array([2.0])})
        assert d1.local_data["y1"] == f1()

    def test_mda(self):
        d1 = AutoPyDiscipline(f1)
        d2 = AutoPyDiscipline(f2)
        mda = create_mda("MDAGaussSeidel", [d1, d2])
        mda.execute()

    def test_fail(self):
        AutoPyDiscipline(f3)
        self.assertRaises(ValueError, AutoPyDiscipline, f4)

        not_a_function = 2
        self.assertRaises(ValueError, AutoPyDiscipline, not_a_function)

    def test_jac_pb(self):
        max_iter = 100
        N = 4
        algo = "L-BFGS-B"

        design_space = create_ds(N)
        pb = OptimizationProblem(design_space)
        pb.objective = MDOFunction(rosen, "rosen", jac=rosen_der)
        execute_algo(pb, algo, max_iter=max_iter)
        fopt_ref = pb.solution.f_opt

        design_space = create_ds(N)
        auto_rosen = AutoPyDiscipline(rosen, rosen_der)
        scn = create_scenario(auto_rosen, "DisciplinaryOpt", "r", design_space)
        scn.execute({"algo": algo, "max_iter": max_iter})
        scn_opt = scn.optimization_result.f_opt

        assert fopt_ref == scn_opt

        auto_rosen = AutoPyDiscipline(rosen)
        self.assertRaises(RuntimeError, auto_rosen._compute_jacobian)
