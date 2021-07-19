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
#       :author : Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import unittest

from numpy import array

from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import SobieskiMission


class TestSobieskiMission(unittest.TestCase):
    """"""

    def setUp(self):
        """At creation of unittest, initiate a sobieski problem class."""
        self.problem = SobieskiProblem("complex128")
        self.threshold = 1e-12

    def test_dweightratio_dwt(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_mission
        indata = self.problem.get_default_inputs_equilibrium(
            names=SobieskiMission().get_input_data_names()
        )
        y_14 = indata["y_14"]
        lin_weightratio = sr.compute_dweightratio_dwt(y_14)
        y_14[0] = y_14[0] + 1j * h
        self.assertAlmostEqual(
            lin_weightratio, sr.compute_weight_ratio(y_14).imag / h, places=8
        )

    def test_dlnweightratio_dwt(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_mission
        indata = self.problem.get_default_inputs_equilibrium(
            names=SobieskiMission().get_input_data_names()
        )
        y_14 = indata["y_14"]
        lin_weightratio = sr.compute_dlnweightratio_dwt(y_14)
        y_14[0] = y_14[0] + 1j * h
        import cmath

        self.assertAlmostEqual(
            lin_weightratio, cmath.log(sr.compute_weight_ratio(y_14)).imag / h, places=8
        )

    def test_d_range_d_wt(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_mission
        indata = self.problem.get_default_inputs_equilibrium(
            names=SobieskiMission().get_input_data_names()
        )
        y_14 = indata["y_14"]
        y_24 = indata["y_24"]
        y_34 = indata["y_34"]
        x_shared = indata["x_shared"]
        sqrt_theta = sr.compute_sqrt_theta(x_shared)
        lin_range = sr.compute_drange_dtotalweight(
            x_shared, y_14, y_24, y_34, sqrt_theta
        )
        y_14[0] = y_14[0] + 1j * h
        self.assertAlmostEqual(
            lin_range, sr.compute_range(x_shared, y_14, y_24, y_34).imag / h, places=8
        )

    def test_d_range_d_wf(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_mission
        indata = self.problem.get_default_inputs_equilibrium(
            names=SobieskiMission().get_input_data_names()
        )
        y_14 = indata["y_14"]
        y_24 = indata["y_24"]
        y_34 = indata["y_34"]
        x_shared = indata["x_shared"]
        sqrt_theta = sr.compute_sqrt_theta(x_shared)
        lin_range = sr.compute_drange_dfuelweight(
            x_shared, y_14, y_24, y_34, sqrt_theta
        )
        y_14[1] = y_14[1] + 1j * h
        self.assertAlmostEqual(
            lin_range, sr.compute_range(x_shared, y_14, y_24, y_34).imag / h, places=8
        )

    def test_jac_mission(self):
        """"""

        sr = SobieskiMission("complex128")
        assert sr.check_jacobian(
            threshold=self.threshold, derr_approx="complex_step", step=1e-30
        )
        inpt_data = {
            "y_24": array([4.16647508]),
            "x_shared": array(
                [
                    5.00000000e-02,
                    4.50000000e04,
                    1.60000000e00,
                    5.50000000e00,
                    5.50000000e01,
                    1.00000000e03,
                ]
            ),
            "y_34": array([1.10754577]),
            "y_14": array([50808.33445658, 7306.20262124]),
        }

        assert sr.check_jacobian(
            inpt_data, derr_approx="complex_step", step=1e-30, threshold=1e-8
        )
