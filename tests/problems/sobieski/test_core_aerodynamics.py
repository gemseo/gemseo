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


from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from builtins import range

from future import standard_library
from numpy import array

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import SobieskiAerodynamics

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class TestSobieskiAerodynamics(unittest.TestCase):
    """ """

    def setUp(self):
        """At creation of unittest, initiate a sobieski problem class"""
        self.problem = SobieskiProblem("complex128")
        self.threshold = 1e-12
        #

    def test_dk_dMach(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"]
        lin_k = sr_aero.compute_dk_aero_dmach(x_shared).real
        x_shared[2] += 1j * h
        self.assertAlmostEqual(
            lin_k, sr_aero.compute_k_aero(x_shared).imag / h, places=12
        )

    def test_dk_dsweep(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"]
        x_shared[1] = 35000.0
        x_shared[4] += 1j * h
        self.assertAlmostEqual(
            sr_aero.compute_dk_aero_dsweep(x_shared).real,
            sr_aero.compute_k_aero(x_shared).imag / h,
            places=12,
        )

    def test_dCDmin_dsweep(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"]
        x_shared[1] = 35000.0
        lin_cd = sr_aero.compute_dcdmin_dsweep(x_shared).real
        x_shared[4] += 1j * h
        Fo1 = 0.95 + 1j * 0
        self.assertAlmostEqual(
            lin_cd, sr_aero.compute_cd_min(x_shared, Fo1).imag / h, places=12
        )

    def test_dCD_dsweep(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"]
        x_shared[1] = 35000.0
        Fo1 = 0.95 + 1j * 0
        CL = 0.0916697016134 + 0j
        Fo2 = 1.00005 + 0j
        lin_cd = sr_aero.compute_dcd_dsweep(x_shared, CL, Fo2).real
        x_shared[4] += 1j * h
        self.assertAlmostEqual(
            lin_cd, sr_aero.compute_cd(x_shared, CL, Fo1, Fo2).imag / h, places=12
        )

    def test_dCD_dMach(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"]
        y_12 = indata["y_12"] * 0.1
        Fo1 = 0.95 + 1j * 0
        CL = sr_aero.compute_cl(x_shared, y_12)
        Fo2 = 1.00005 + 0j
        lin_cd = sr_aero.compute_dcd_dmach(x_shared, y_12, Fo2).real
        x_shared[2] += 1j * h
        self.assertAlmostEqual(
            lin_cd, sr_aero.compute_cd(x_shared, CL, Fo1, Fo2).imag / h, places=4
        )

    def test_dCD_dsref(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"]
        y_12 = indata["y_12"]
        Fo1 = 0.95 + 1j * 0
        Fo2 = 1.00005 + 0j
        lin_cd = sr_aero.compute_dcd_dsref(x_shared, y_12, Fo2)
        x_shared[5] += 1j * h
        CL = sr_aero.compute_cl(x_shared, y_12)
        self.assertAlmostEqual(
            lin_cd, sr_aero.compute_cd(x_shared, CL, Fo1, Fo2).imag / h, places=12
        )

    def test_dCL_dh(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"]
        y_12 = indata["y_12"]
        y_12[0] = 5.06069742e04 + 0.0j
        lin_cl = sr_aero.compute_dcl_dh(x_shared, y_12).real
        x_shared[1] += 1j * h
        self.assertAlmostEqual(
            lin_cl, sr_aero.compute_cl(x_shared, y_12).imag / h, places=12
        )

        sr_aero = self.problem.sobieski_aerodynamics
        x_shared[1] = 35000.0
        lin_cl = sr_aero.compute_dcl_dh(x_shared, y_12).real
        x_shared[1] += 1j * h
        self.assertAlmostEqual(
            lin_cl, sr_aero.compute_cl(x_shared, y_12).imag / h, places=12
        )

    def test_dCL_dsref(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"]
        y_12 = indata["y_12"]
        y_12[0] = 5.06069742e04 + 0.0j
        lin_cl = sr_aero.compute_dcl_dsref(x_shared, y_12)
        x_shared[5] += 1j * h
        self.assertAlmostEqual(
            lin_cl, sr_aero.compute_cl(x_shared, y_12).imag / h, places=12
        )

    def test_dCL_dMach(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"][:]
        y_12 = indata["y_12"]
        y_12[0] = 5.06069742e04 + 0.0j
        lin_cl = sr_aero.compute_dcl_dmach(x_shared, y_12).real
        x_shared[2] += 1j * h
        self.assertAlmostEqual(
            lin_cl, sr_aero.compute_cl(x_shared, y_12).imag / h, places=12
        )

        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        x_shared = indata["x_shared"][:]
        x_shared[1] = 35000.0
        lin_cl = sr_aero.compute_dcl_dmach(x_shared, y_12).real
        x_shared[2] += 1j * h
        self.assertAlmostEqual(
            lin_cl, sr_aero.compute_cl(x_shared, y_12).imag / h, places=12
        )

    def test_drhoV2_dh(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"]
        lin_rhoV2 = sr_aero.compute_drhov2_dh(x_shared).real
        x_shared[1] += 1j * h
        self.assertAlmostEqual(
            lin_rhoV2, sr_aero.compute_rhov2(x_shared).imag / h, places=12
        )

        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        x_shared[1] = 35000.0
        lin_rhoV2 = sr_aero.compute_drhov2_dh(x_shared).real
        x_shared[1] += 1j * h
        self.assertAlmostEqual(
            lin_rhoV2, sr_aero.compute_rhov2(x_shared).imag / h, places=12
        )

    def test_drhoV2_dM(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        h = 1e-30
        x_shared = indata["x_shared"]
        lin_rhoV2 = sr_aero.compute_drhov2_dmach(x_shared).real
        x_shared[2] += 1j * h
        self.assertAlmostEqual(
            lin_rhoV2, sr_aero.compute_rhov2(x_shared).imag / h, places=12
        )

        sr_aero = self.problem.sobieski_aerodynamics
        indata = self.problem.get_default_inputs(
            names=SobieskiAerodynamics().get_input_data_names()
        )
        x_shared = indata["x_shared"]
        x_shared[1] = 35000.0
        lin_rhoV2 = sr_aero.compute_drhov2_dmach(x_shared).real
        x_shared[2] += 1j * h
        self.assertAlmostEqual(
            lin_rhoV2, sr_aero.compute_rhov2(x_shared).imag / h, places=12
        )

    def test_dv_dMach(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        h = 1e-30
        mach = 1.8
        altitude = 45000.0
        lin_V = sr_aero.compute_dv_dmach(altitude)
        self.assertAlmostEqual(
            lin_V, sr_aero.compute_rho_v(mach + 1j * h, altitude)[1].imag / h, places=12
        )

        sr_aero = self.problem.sobieski_aerodynamics
        h = 1e-30
        mach = 1.4
        altitude = 35000.0
        lin_V = sr_aero.compute_dv_dmach(altitude)
        self.assertAlmostEqual(
            lin_V, sr_aero.compute_rho_v(mach + 1j * h, altitude)[1].imag / h, places=12
        )

    def test_dV_dh_drho_dh(self):
        """ """
        sr_aero = self.problem.sobieski_aerodynamics
        h = 1e-30
        mach = 1.6
        altitude = 45000.0
        dVdh_drhodh_ref = array(sr_aero.compute_rho_v(mach, altitude + 1j * h)).imag / h
        dVdh_ref = dVdh_drhodh_ref[0]
        drhodh_ref = dVdh_drhodh_ref[1]

        dVdh_drhodh = array(sr_aero.compute_drho_dh_dv_dh(mach, altitude))
        dVdh = dVdh_drhodh[0].real
        drhodh = dVdh_drhodh[1].real

        self.assertAlmostEqual(dVdh_ref, dVdh, places=4)
        self.assertAlmostEqual(drhodh_ref, drhodh, places=4)

        altitude = 35000.0
        dVdh_drhodh_ref = array(sr_aero.compute_rho_v(mach, altitude + 1j * h)).imag / h
        dVdh_ref = dVdh_drhodh_ref[0]
        drhodh_ref = dVdh_drhodh_ref[1]

        dVdh_drhodh = array(sr_aero.compute_drho_dh_dv_dh(mach, altitude))
        dVdh = dVdh_drhodh[0].real
        drhodh = dVdh_drhodh[1].real
        self.assertAlmostEqual(dVdh_ref, dVdh, places=4)
        self.assertAlmostEqual(drhodh_ref, drhodh, places=4)

    #

    def test_jac_aero(self):
        """ """
        sr = SobieskiAerodynamics("complex128")
        indata = self.problem.get_default_inputs(names=sr.get_input_data_names())
        assert sr.check_jacobian(
            indata, threshold=self.threshold, derr_approx="complex_step", step=1e-30
        )
        #
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
