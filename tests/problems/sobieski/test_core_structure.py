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
#      :author: Damien Guenot - 18 mars 2016
#    OTHER AUTHORS   - MACROSCOPIC CHANGES


from __future__ import division, unicode_literals

import unittest

from numpy import array

from gemseo.problems.sobieski.base import SobieskiBase
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.core_structure import SobieskiStructure as CoreStructure
from gemseo.problems.sobieski.wrappers import SobieskiStructure


class TestSobieskiStructure(unittest.TestCase):
    """"""

    def setUp(self):
        """At creation of unittest, initiate a sobieski problem class."""
        self.problem = SobieskiProblem("complex128")
        self.threshold = 1e-12

    def test_dfuelweightdtoverc(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_structure
        indata = self.problem.get_default_inputs(
            names=SobieskiStructure().get_input_data_names()
        )
        x_shared = indata["x_shared"]
        lin_wf = sr.compute_dfuelwing_dtoverc(x_shared)
        x_shared[0] += 1j * h
        self.assertAlmostEqual(
            lin_wf, sr.compute_fuelwing_weight(x_shared).imag / h, places=8
        )

    def test_dfuelweightd_ar(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_structure
        indata = self.problem.get_default_inputs(
            names=SobieskiStructure().get_input_data_names()
        )
        x_shared = indata["x_shared"]
        lin_wf = sr.compute_dfuelwing_dar(x_shared)
        x_shared[3] += 1j * h
        self.assertAlmostEqual(
            lin_wf, sr.compute_fuelwing_weight(x_shared).imag / h, places=8
        )

    def test_dfuelweightdsref(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_structure
        indata = self.problem.get_default_inputs(
            names=SobieskiStructure().get_input_data_names()
        )
        x_shared = indata["x_shared"]
        lin_wf = sr.compute_dfuelwing_dsref(x_shared)
        x_shared[5] += 1j * h
        self.assertAlmostEqual(
            lin_wf, sr.compute_fuelwing_weight(x_shared).imag / h, places=8
        )

    def test_jac_structure(self):
        """"""

        sr = SobieskiStructure("complex128")
        indata = self.problem.get_default_inputs(names=sr.get_input_data_names())
        self.assertTrue(
            sr.check_jacobian(
                indata, threshold=self.threshold, derr_approx="complex_step", step=1e-30
            )
        )

        indata = self.problem.get_default_inputs_feasible(
            names=sr.get_input_data_names()
        )
        assert sr.check_jacobian(
            indata, threshold=self.threshold, derr_approx="complex_step", step=1e-30
        )

        indata = self.problem.get_default_inputs_equilibrium(
            names=sr.get_input_data_names()
        )
        assert sr.check_jacobian(
            indata, threshold=self.threshold, derr_approx="complex_step", step=1e-30
        )

        for _ in range(5):
            indata = self.problem.get_random_input(
                names=sr.get_input_data_names(), seed=1
            )
            assert sr.check_jacobian(
                indata, threshold=self.threshold, derr_approx="complex_step", step=1e-30
            )
        core_s = CoreStructure(SobieskiBase("complex128"))
        core_s.derive_constraints(
            sr.jac, indata["x_shared"], indata["x_1"], indata["y_21"], true_cstr=True
        )

    def test_jac2_sobieski_struct(self):

        inpt_data = {
            "y_31": array([6555.68459235 + 0j]),
            "y_21": array([50606.9742 + 0j]),
            "x_shared": array(
                [
                    5.00000000e-02 + 0j,
                    4.50000000e04 + 0j,
                    1.60000000e00 + 0j,
                    5.50000000e00 + 0j,
                    5.50000000e01 + 0j,
                    1.00000000e03 + 0j,
                ]
            ),
            "x_1": array([0.25 + 0j, 1.0 + 0j]),
        }

        st = SobieskiStructure("complex128")
        assert st.check_jacobian(
            inpt_data, threshold=1e-8, derr_approx="complex_step", step=1e-30
        )
