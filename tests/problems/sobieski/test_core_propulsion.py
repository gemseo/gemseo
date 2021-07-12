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

from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import SobieskiPropulsion


class TestSobieskiPropulsion(unittest.TestCase):
    """"""

    def setUp(self):
        """At creation of unittest, initiate a sobieski problem class."""
        self.problem = SobieskiProblem("complex128")
        self.threshold = 1e-12

    def test_d_esf_ddrag(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_propulsion
        indata = self.problem.get_default_inputs(
            names=SobieskiPropulsion().get_input_data_names()
        )
        drag = indata["y_23"][0]
        throttle = indata["x_3"][0]
        lin_esf = sr.compute_desf_ddrag(throttle)
        drag = drag + 1j * h
        self.assertAlmostEqual(
            lin_esf, sr.compute_esf(drag, throttle).imag / h, places=8
        )

    def test_d_esf_dthrottle(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_propulsion
        indata = self.problem.get_default_inputs(
            names=SobieskiPropulsion().get_input_data_names()
        )
        drag = indata["y_23"][0]
        throttle = indata["x_3"][0]
        lin_esf = sr.compute_desf_dthrottle(drag, throttle)
        throttle = throttle + 1j * h
        self.assertAlmostEqual(
            lin_esf, sr.compute_esf(drag, throttle).imag / h, places=4
        )

    def test_blackbox_propulsion(self):
        indata = self.problem.get_default_inputs(names=["x_shared", "y_23", "x_3"])
        x_shared = indata["x_shared"]
        y_23 = indata["y_23"]
        x_3 = indata["x_3"]
        _, _, _, _, g_3 = self.problem.sobieski_propulsion.blackbox_propulsion(
            x_shared, y_23, x_3, true_cstr=False
        )

        _, _, _, _, g_3_t = self.problem.sobieski_propulsion.blackbox_propulsion(
            x_shared, y_23, x_3, true_cstr=True
        )

        assert len(g_3) == len(g_3_t) + 1
        jac = self.problem.sobieski_propulsion._SobieskiPropulsion__initialize_jacobian(
            False
        )

        jac_t = (
            self.problem.sobieski_propulsion._SobieskiPropulsion__initialize_jacobian(
                True
            )
        )

        for var in ["x_shared", "x_3"]:
            assert jac["g_3"][var].shape[0] == jac_t["g_3"][var].shape[0] + 1

        self.problem.sobieski_propulsion.derive_blackbox_propulsion(
            x_shared, y_23, x_3, true_cstr=True
        )

    def test_d_we_dthrottle(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_propulsion
        indata = self.problem.get_default_inputs(
            names=SobieskiPropulsion().get_input_data_names()
        )
        drag = indata["y_23"][0]
        throttle = indata["x_3"][0]
        d_esf_dthrottle = sr.compute_desf_dthrottle(drag, throttle)
        esf = sr.compute_esf(drag, throttle)
        lin_we = sr.compute_dengineweight_dvar(esf, d_esf_dthrottle)

        throttle = throttle + 1j * h
        esf = sr.compute_esf(drag, throttle)
        self.assertAlmostEqual(lin_we, sr.compute_engine_weight(esf).imag / h, places=8)

    def test_d_we_ddrag(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_propulsion
        indata = self.problem.get_default_inputs(
            names=SobieskiPropulsion().get_input_data_names()
        )
        drag = indata["y_23"][0]
        throttle = indata["x_3"][0]
        d_esf_ddrag = sr.compute_desf_ddrag(throttle)
        esf = sr.compute_esf(drag, throttle)
        lin_we = sr.compute_dengineweight_dvar(esf, d_esf_ddrag)

        drag = drag + 1j * h
        esf = sr.compute_esf(drag, throttle)
        self.assertAlmostEqual(lin_we, sr.compute_engine_weight(esf).imag / h, places=8)

    #

    def test_d_sfc_dthrottle(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_propulsion
        indata = self.problem.get_default_inputs(
            names=SobieskiPropulsion().get_input_data_names()
        )
        x_shared = indata["x_shared"]
        throttle = indata["x_3"][0]

        lin_sfc = sr.compute_dsfc_dthrottle(x_shared, throttle)
        throttle = throttle + 1j * h
        self.assertAlmostEqual(
            lin_sfc, sr.compute_sfc(x_shared, throttle).imag / h, places=8
        )

    #

    def test_d_sfc_dh(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_propulsion
        indata = self.problem.get_default_inputs(
            names=SobieskiPropulsion().get_input_data_names()
        )
        throttle = indata["x_3"][0]

        x_shared = indata["x_shared"]
        lin_sfc = sr.compute_dsfc_dh(x_shared, throttle)

        x_shared[1] = x_shared[1] + 1j * h
        self.assertAlmostEqual(
            lin_sfc, sr.compute_sfc(x_shared, throttle).imag / h, places=8
        )

    #

    def test_d_sfc_d_m(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_propulsion
        indata = self.problem.get_default_inputs(
            names=SobieskiPropulsion().get_input_data_names()
        )
        throttle = indata["x_3"][0] * 16168.6

        x_shared = indata["x_shared"]
        lin_sfc = sr.compute_dsfc_dmach(x_shared, throttle)

        x_shared[2] = x_shared[2] + 1j * h
        self.assertAlmostEqual(
            lin_sfc, sr.compute_sfc(x_shared, throttle).imag / h, places=8
        )

    #

    def test_dthrottle_constraint_dthrottle(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_propulsion
        indata = self.problem.get_default_inputs(
            names=SobieskiPropulsion().get_input_data_names()
        )
        x_shared = indata["x_shared"]
        x_3 = indata["x_3"]
        lin_throttle = sr.compute_dthrconst_dthrottle(x_shared)
        x_3[0] = x_3[0] + 1j * h
        self.assertAlmostEqual(
            lin_throttle,
            sr.compute_throttle_constraint(x_shared, x_3[0]).imag / h,
            places=8,
        )

    def test_dthrottle_constraint_dh(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_propulsion
        indata = self.problem.get_default_inputs(
            names=SobieskiPropulsion().get_input_data_names()
        )
        x_shared = indata["x_shared"]
        x_3 = indata["x_3"]
        lin_throttle = sr.compute_dthrcons_dh(x_shared, x_3[0])
        x_shared[1] = x_shared[1] + 1j * h
        self.assertAlmostEqual(
            lin_throttle,
            sr.compute_throttle_constraint(x_shared, x_3[0]).imag / h,
            places=8,
        )

    def test_dthrottle_constraint_dmach(self):
        """"""
        h = 1e-30
        sr = self.problem.sobieski_propulsion
        indata = self.problem.get_default_inputs(
            names=SobieskiPropulsion().get_input_data_names()
        )
        x_shared = indata["x_shared"]
        x_3 = indata["x_3"]
        lin_throttle = sr.compute_dthrconst_dmach(x_shared, x_3[0])
        x_shared[2] = x_shared[2] + 1j * h
        self.assertAlmostEqual(
            lin_throttle,
            sr.compute_throttle_constraint(x_shared, x_3[0]).imag / h,
            places=8,
        )

    def test_jac_prop(self):
        """"""
        sr = SobieskiPropulsion("complex128")
        indata = self.problem.get_default_inputs(names=sr.get_input_data_names())
        assert sr.check_jacobian(
            indata, threshold=self.threshold, derr_approx="complex_step", step=1e-30
        )
        #
        indata = self.problem.get_default_inputs_feasible(
            names=sr.get_input_data_names()
        )
        assert sr.check_jacobian(indata, derr_approx="complex_step", step=1e-30)

        indata = self.problem.get_default_inputs_equilibrium(
            names=sr.get_input_data_names()
        )
        assert sr.check_jacobian(
            indata, threshold=self.threshold, derr_approx="complex_step", step=1e-30
        )
        ##
        indata = self.problem.get_default_inputs_equilibrium(
            names=sr.get_input_data_names()
        )

        for _ in range(5):
            indata = self.problem.get_random_input(
                names=sr.get_input_data_names(), seed=1
            )
            assert sr.check_jacobian(
                indata, threshold=self.threshold, derr_approx="complex_step", step=1e-30
            )
