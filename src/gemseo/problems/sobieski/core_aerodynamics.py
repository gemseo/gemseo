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
#    INITIAL AUTHORS - initial API and implementation
#               and/or initial documentation
#        :author: Sobieski, Agte, and Sandusky
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Damien Guenot
#        :author: Francois Gallard
# From NASA/TM-1998-208715
# Bi-Level Integrated System Synthesis (BLISS)
# Sobieski, Agte, and Sandusky
"""
SSBJ Aerodynamics computation
*****************************
"""
from __future__ import division, unicode_literals

import logging
from math import pi

from numpy import array, atleast_2d, cos, sin, sqrt, zeros

LOGGER = logging.getLogger(__name__)
DEG_TO_RAD = pi / 180.0


class SobieskiAerodynamics(object):
    """Class defining aerodynamical analysis for Sobieski problem and related method to
    the aerodynamical problem such as disciplines computation, constraints, reference
    optimum."""

    DTYPE_COMPLEX = "complex128"
    DTYPE_DOUBLE = "float64"

    PRESSURE_GRADIENT_LIMIT = 1.04

    def __init__(self, sobieski_base):
        """Constructor."""
        self.base = sobieski_base
        self.constants = self.base.default_constants()
        (
            self.x_initial,
            self.tc_initial,
            self.half_span_initial,
            self.aero_center_initial,
            self.cf_initial,
            self.mach_initial,
            self.h_initial,
            self.throttle_initial,
            self.lift_initial,
            self.twist_initial,
            self.esf_initial,
        ) = self.base.get_initial_values()
        self.dtype = self.base.dtype
        self.math = self.base.math

    def __set_coeff_drag_f1(self, y_32, x_2):
        """Setting of flags used for determination of polynomial coefficients.

        :param y_32: shared variables coming from blackbox_propulsion

            - y_32[0]: engine scale factor

        :type y_32: numpy array
        :param x_2: aero. design variable

            - x_2[0]: friction coeff

        :type x_2: numpy array
        :returns: s_initial, S_new, flag, bound

            - s_initial: reference value
            - s_new: new values

        :rtype: numpy array
        """
        s_initial = array([self.esf_initial, self.cf_initial], dtype=self.dtype)
        s_new = array([y_32[0], x_2[0]], dtype=self.dtype)
        flag = array([1, 1], dtype=self.dtype)
        bound = array([0.25, 0.25], dtype=self.dtype)
        return s_initial, s_new, flag, bound

    def __set_coeff_drag_f2(self, y_12):
        """Setting of flags used for determination of polynomial coefficients.

        :param y_12: shared variables coming from blackbox_structure
        :type y_12: numpy array
        :returns: s_initial, s_new, flag, bound

                - s_initial: reference value
                - s_new: new values

        :rtype: numpy array
        """
        s_initial = array([self.twist_initial], dtype=self.dtype)
        s_new = array([y_12[1]], dtype=self.dtype)
        flag = array([5], dtype=self.dtype)
        bound = array([0.25], dtype=self.dtype)
        return s_initial, s_new, flag, bound

    def __set_coeff_drag_f3(self, x_shared):
        """Setting of flags used for determination of polynomial coefficients.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: s_initial, s_new, flag, bound

                - s_initial: reference value
                - s_new: new values

        :rtype: numpy array
        """
        s_initial = array([self.tc_initial], dtype=self.dtype)
        s_new = array([x_shared[0]], dtype=self.dtype)
        flag = array([1], dtype=self.dtype)
        bound = array([0.25], dtype=self.dtype)
        return s_initial, s_new, flag, bound

    def compute_k_aero(self, x_shared):
        """Computation of a induced drag coefficient (related to lift)

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: k_aero
        :rtype: numpy array
        """
        #        if x_shared[2] >= 1:  # Mach
        #            k = (x_shared[2] ** 2 - 1) * \
        #                self.math.cos(x_shared[4] * DEG_TO_RAD) / \
        #                (4.0 * self.math.sqrt(x_shared[4] ** 2 - 1) - 2)
        #        else:
        #            k = 1. / (self.math.pi * 0.8 * x_shared[3])
        k_aero = (
            (x_shared[2] ** 2 - 1)
            * self.math.cos(x_shared[4] * DEG_TO_RAD)
            / (4.0 * self.math.sqrt(x_shared[4] ** 2 - 1) - 2)
        )
        return k_aero

    #     @staticmethod
    #     def compute_dk_aero_dAR(self, x_shared):
    #         #        if x_shared[2] >= 1:
    #         #            dk_dAR = 0.0
    #         #        else:
    #         #            dk_dAR = -1.0 / (0.8 * self.math.pi *
    # abs(x_shared[3])**2)
    #         dk_dAR = 0.0
    #         return dk_dAR

    @staticmethod
    def compute_dk_aero_dsweep(x_shared):
        """Computation of a derivative of k_aero wrt sweep.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: dk_aero_dsweep
        :rtype: numpy array
        """
        #        if x_shared[2] >= 1:
        #            dk_dsweep = -DEG_TO_RAD * (x_shared[2] ** 2 - 1) * \
        #                self.math.sin(x_shared[4] * DEG_TO_RAD) / \
        #                (4.0 * self.math.sqrt(x_shared[4] ** 2 - 1) - 2)
        #        else:
        #            dk_dsweep = 0.0
        u_velo = (x_shared[2] * x_shared[2] - 1.0) * cos(x_shared[4] * DEG_TO_RAD)
        up_velo = (
            -DEG_TO_RAD
            * (x_shared[2] * x_shared[2] - 1.0)
            * sin(x_shared[4] * DEG_TO_RAD)
        )
        v_velo = 4.0 * sqrt(x_shared[4] * x_shared[4] - 1.0) - 2.0
        vp_velo = 4.0 * x_shared[4] * (x_shared[4] * x_shared[4] - 1.0) ** -0.5
        dk_aero_dsweep = (up_velo * v_velo - u_velo * vp_velo) / v_velo ** 2
        return dk_aero_dsweep

    @staticmethod
    def compute_dk_aero_dmach(x_shared):
        """Computation of a derivative of k_aero wrt Mach.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: dk_aero_dmach
        :rtype: numpy array
        """
        # if x_shared[2] >= 1:
        #     dk_dM = abs(x_shared[3]) * (2.0 * abs(x_shared[2])) * \
        #             np.cos(x_shared[4] * np.pi / 180.) /
        # (4. * abs(x_shared[3]) * \
        #             np.sqrt(abs(x_shared[4]**2 - 1.) - 2.))
        # else:
        #     dk_dM = 0.0
        dk_aero_dmach = (
            (2.0 * x_shared[2])
            * cos(x_shared[4] * pi / 180.0)
            / (4.0 * sqrt(x_shared[4] ** 2 - 1.0) - 2.0)
        )
        return dk_aero_dmach

    def compute_dadimcf_dcf(self, x_2):
        """Computation of a derivative of adim friction coeff of polynomial wrt friction
        coeff.

        :param x_2: local design variables
        :type x_2: numpy array
        :returns: derivative of adim friction coeff of polynomial
             wrt friction coeff
        :rtype: numpy array
        """
        dadimcf_dcf = self.base.derive_normalize_s(self.cf_initial, x_2[0])
        return dadimcf_dcf

    def compute_dadimtwist_dtwist(self, y_12):
        """Computation of a derivative of adim twist of polynomial wrt twist.

        :param y_12: coupling variable from weight analysis
        :type y_12: numpy array
        :returns: derivative of adim twist of polynomial wrt twist
        :rtype: numpy array
        """
        dadimtwist_dtwist = self.base.derive_normalize_s(self.twist_initial, y_12[1])
        return dadimtwist_dtwist

    def compute_dadimesf_desf(self, y_32):
        """Computation of a derivative of adim ESF of polynomial wrt ESF.

        :param y_32: coupling variable from propulsion analysis
        :type y_32: numpy array
        :returns: derivative of adim ESF of polynomial wrt ESF
        :rtype: numpy array
        """
        return self.base.derive_normalize_s(self.esf_initial, y_32[0])

    def compute_dadimtaper_dtaper(self, x_shared):
        """Computation of a derivative of adim taper-ratio of polynomial wrt taper-
        ratio.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: derivative of adim taper-ratio of polynomial wrt taper-ratio
        :rtype: numpy array
        """
        return self.base.derive_normalize_s(self.tc_initial, x_shared[0])

    def compute_cd_min(self, x_shared, fo1):
        """Computation of a 2D minimum drag coefficient.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :param fo1: coefficient for engine size
        :type fo1: numpy array
        :returns: CDmin : drag coefficient
        :rtype: numpy array
        """
        cdmin_incomp = self.constants[4]
        return (
            cdmin_incomp * fo1
            + 3.05
            * x_shared[0] ** (5.0 / 3.0)
            * (self.math.cos(x_shared[4] * DEG_TO_RAD)) ** 1.5
        )

    @staticmethod
    def compute_dcdmin_dsweep(x_shared):
        """Computation of derivative of 2D minimum drag coefficient wrt sweep.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: dCDmin_dsweep
        :rtype: numpy array
        """
        ang_rad = x_shared[4] * DEG_TO_RAD
        return (
            -3.05
            * 1.5
            * x_shared[0] ** (5.0 / 3.0)
            * cos(ang_rad) ** 0.5
            * DEG_TO_RAD
            * sin(ang_rad)
        )

    def compute_cd(self, x_shared, lift_coeff, fo1, fo2):
        """Computation of total drag coefficient.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :param lift_coeff: lift coefficient
        :type lift_coeff: numpy array
        :param fo1: coefficient for engine size
        :type fo1: numpy array
        :param fo2: coefficient for twist influence on drag
        :type fo2: numpy array
        :returns: drag_coefficient : drag coefficient
        :rtype: numpy array
        """
        k_aero = self.compute_k_aero(x_shared)
        cdmin = self.compute_cd_min(x_shared, fo1)
        return fo2 * (cdmin + k_aero * lift_coeff * lift_coeff)

    def compute_drag(self, drag_coeff, x_shared):
        """Computation of drag from drag coefficient.

        :param drag_coeff: lift coefficient
        :type drag_coeff: numpy array
        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: drag as a force
        :rtype: numpy array
        """
        return self.compute_force(drag_coeff, x_shared[5], x_shared[2], x_shared[1])

    def compute_dcd_dsweep(self, x_shared, lift_coeff, fo2):
        """Computation of derivative of drag coefficient wrt sweep.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :param lift_coeff: lift coefficient
        :type lift_coeff: numpy array
        :param fo2: coefficient for twist influence on drag
        :type fo2: numpy array
        :returns: dCd/dsweep
        :rtype: numpy array
        """
        dcdmin_dsweep = self.compute_dcdmin_dsweep(x_shared)
        dk_dsweep = self.compute_dk_aero_dsweep(x_shared)
        return fo2 * (dcdmin_dsweep + lift_coeff * lift_coeff * dk_dsweep)

    def compute_dcd_dsref(self, x_shared, y_12, fo2):
        """Computation of derivative of drag coefficient wrt sweep.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :param y_12: lift coefficient
        :type y_12: numpy array
        :param fo2: coefficient for twist influence on drag
        :type fo2: numpy array
        :returns: dCd/dsweep
        :rtype: numpy array
        """
        k_aero = self.compute_k_aero(x_shared)
        lift_coeff = self.compute_cl(x_shared, y_12)
        dcl_dsref = self.compute_dcl_dsref(x_shared, y_12)
        return 2 * k_aero * lift_coeff * dcl_dsref * fo2

    def compute_dcd_dmach(self, x_shared, y_12, fo2):
        """Computation of derivative of drag coefficient wrt Mach.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :param y_12: coupling variable from weight analysis
        :type y_12: numpy array
        :param fo2: coefficient for twist influence on drag
        :type fo2: numpy array
        :returns: dcd_dmach
        :rtype: numpy array
        """
        k_aero = self.compute_k_aero(x_shared)
        dk_dmach = self.compute_dk_aero_dmach(x_shared)
        lift_coeff = self.compute_cl(x_shared, y_12)
        dcl_dmach = self.compute_dcl_dmach(x_shared, y_12)
        return (2.0 * k_aero * dcl_dmach + lift_coeff * dk_dmach) * lift_coeff * fo2

    def compute_cl(self, x_shared, y_12):
        """Computation of lift coefficient.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :param y_12: coupling variable from weight analysis
        :type y_12: numpy array
        :returns: lift coefficient Cl
        :rtype: numpy array
        """
        return self.compute_adim_coeff(y_12[0], x_shared[5], x_shared[2], x_shared[1])

    def compute_dcl_dh(self, x_shared, y_12):
        """Computation of derivative of lift coefficient wrt altitude.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :param y_12: coupling variable from weight analysis
        :type y_12: numpy array
        :returns: derivative of lift coefficient wrt altitude
        :rtype: numpy array
        """
        rhov2 = self.compute_rhov2(x_shared)
        drhov2_dh = self.compute_drhov2_dh(x_shared)
        return -2 * y_12[0] / x_shared[5] * drhov2_dh / (rhov2 ** 2)

    def compute_dcl_dmach(self, x_shared, y_12):
        """Computation of derivative of lift coefficient wrt reference surface.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :param y_12: coupling variable from weight analysis
        :type y_12: numpy array
        :returns: derivative of lift coefficient wrt reference surface
        :rtype: numpy array
        """
        rho, velocity = self.compute_rho_v(x_shared[2], x_shared[1])
        dv_dmach = self.compute_dv_dmach(x_shared[1])
        dcl_dmach = (
            -4.0
            * y_12[0]
            * dv_dmach
            / (rho * x_shared[5] * velocity * velocity * velocity)
        )
        return dcl_dmach

    def compute_dcl_dsref(self, x_shared, y_12):
        """Computation of derivative of lift coefficient wrt Mach number.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :param y_12: coupling variable from weight analysis
        :type y_12: numpy array
        :returns: derivative of lift coefficient wrt Mach number
        :rtype: numpy array
        """
        return -self.compute_cl(x_shared, y_12) / x_shared[5]

    def compute_rhov2(self, x_shared):
        """Computation of rho * velocity * velocity (2*dynamic pressure)

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: rho * velocity * velocity
        :rtype: numpy array
        """
        rho, velocity = self.compute_rho_v(x_shared[2], x_shared[1])
        return rho * velocity * velocity

    def compute_drhov2_dh(self, x_shared):
        """Computation of derivative of rhoV**2 wrt altitude.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: drhov2_dh (derivative of rhoV**2 wrt altitude)
        :rtype: numpy array
        """
        rho, velocity = self.compute_rho_v(x_shared[2], x_shared[1])
        drho_dh, dv_dh = self.compute_drho_dh_dv_dh(x_shared[2], x_shared[1])
        return drho_dh * velocity * velocity + 2.0 * rho * dv_dh * velocity

    def compute_drhov2_dmach(self, x_shared):
        """Computation of derivative of rhoV**2 wrt Mach.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: drhov2_dmach (derivative of rhoV**2 wrt Mach)
        :rtype: numpy array
        """
        rho, velocity = self.compute_rho_v(x_shared[2], x_shared[1])
        dv_dmach = self.compute_dv_dmach(x_shared[1])
        return 2.0 * rho * dv_dmach * velocity

    def compute_rho_v(self, mach, altitude):
        """From given Mach number and altitude, compute velocity and density.

        :param mach: Mach number
        :type mach: float
        :param altitude: altitude
        :type altitude: float
        :returns: rho, velocity : density and velocity
        :rtype: float
        """
        if altitude.real < 36089.0:
            velocity = mach * 1116.39 * self.math.sqrt(1 - 6.875e-6 * altitude)
            rho = 2.377e-3 * (1 - 6.875e-6 * altitude) ** 4.2561
        else:
            velocity = mach * 968.1
            rho = 2.377e-3 * 0.2971 * self.math.exp((36089.0 - altitude) / 20806.7)
        return rho, velocity

    def compute_dv_dmach(self, altitude):
        """Computation of derivative of velocity wrt Mach.

        :param altitude: altitude
        :type altitude: numpy array
        :returns: drhov2_dmach (derivative of rhoV**2 wrt Mach)
        :rtype: numpy array
        """
        if altitude.real < 36089.0:
            dv_dmach = 1116.39 * self.math.sqrt(1 - 6.875e-6 * altitude)
        else:
            dv_dmach = 968.1
        return dv_dmach

    def compute_drho_dh_dv_dh(self, mach, altitude):
        """Compute derivative of velocity and density wrt altitude.

        :param mach: Mach number
        :type mach: float
        :param altitude: altitude
        :type altitude: float
        :returns: drho_dh, dv_dh
        :rtype: float
        """
        if altitude.real <= 36089.0:
            drho_dh = (
                -2.377e-3 * 4.2561 * 6.875e-6 * (1.0 - 6.875e-6 * altitude) ** 3.2561
            )
            dv_dh = (
                -6.875e-6 * 1116.39 * mach * 0.5 * (1.0 - 6.875e-6 * altitude) ** -0.5
            )
        else:
            drho_dh = (
                -2.377e-3
                * 0.2971
                / 20806.7
                * self.math.exp((36089.0 - altitude) / 20806.7)
            )
            dv_dh = 0.0
        return drho_dh, dv_dh

    def compute_adim_coeff(self, force, sref, mach, altitude):
        """Compute adim. force coeff from force (lift or drag)

        :param force: Force
        :type force: float
        :param sref: reference surface
        :type sref: float
        :param mach: Mach number
        :type mach: float
        :param altitude: altitude
        :type altitude: float
        :returns: force coefficient: Cl or Cd according to input force
        :rtype: float
        """
        rho, velocity = self.compute_rho_v(mach, altitude)
        return force / (0.5 * rho * velocity * velocity * sref)

    def compute_force(self, adim_coeff, sref, mach, altitude):
        """Compute force (lift or drag) from adim coeff.

        :param adim_coeff: force coefficient
        :type adim_coeff: float
        :param sref: reference surface
        :type sref: float
        :param mach: Mach number
        :type mach: float
        :param altitude: altitude
        :type altitude: float
        :returns: force: lift or drag
        :rtype: float
        """
        rho, velocity = self.compute_rho_v(mach, altitude)
        #        return adim_coeff * 0.5 * rho * V * V * sref
        return 0.5 * rho * velocity * velocity * adim_coeff * sref

    def blackbox_aerodynamics(self, x_shared, y_12, y_32, x_2, true_cstr=False):
        """This function calculates drag and lift to drag ratio of A/C.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: numpy array
        :param y_12: shared variables coming from blackbox_structure:

            - y_12[0]: total aircraft weight
            - y_12[1]: wing twist

        :type y_12: numpy array
        :param y_32: shared variables coming from blackbox_propulsion:

            - y_32[0]: engine scale factor

        :type y_32: numpy array
        :param x_2: aero. design variable:

            - x_2[0]: friction coeff

        :type x_2: numpy array
        :param true_cstr: Default value = False)
        :returns: y_2, y_21, y_23, y_24, g_2

            - y_2: aero. analysis outputs

                - y_2[0]: lift
                - y_2[1]: drag
                - y_2[2]: lift/drag ratio

            - y_21: shared variable for blackbox_structure (lift)
            - y_23: shared variable for blackbox_propulsion (drag)
            - y_24: shared variable for BlackBoxMission (lift/drag ratio)
            - g_2: aero constraint (pressure gradient)

        :rtype: numpy array, numpy array, numpy array, numpy array, numpy array
        """

        y_2 = zeros(3, dtype=self.dtype)
        y_23 = zeros(1, dtype=self.dtype)
        y_24 = zeros(1, dtype=self.dtype)
        y_21 = zeros(1, dtype=self.dtype)
        g_2 = zeros(1, dtype=self.dtype)

        lift_coeff = self.compute_cl(x_shared, y_12)

        # Modification of CDmin for ESF and Cf
        s_initial1, s_new1, flag1, bound1 = self.__set_coeff_drag_f1(y_32, x_2)
        fo1 = self.base.poly_approx(s_initial1, s_new1, flag1, bound1)

        # Modification of drag_coeff for wing twist
        s_initial2, s_new2, flag2, bound2 = self.__set_coeff_drag_f2(y_12)
        fo2 = self.base.poly_approx(s_initial2, s_new2, flag2, bound2)

        drag_coeff = self.compute_cd(x_shared, lift_coeff, fo1, fo2)
        drag = self.compute_drag(drag_coeff, x_shared)

        y_2[1] = drag
        y_2[2] = lift_coeff / drag_coeff
        y_2[0] = y_12[0]
        y_23[0] = y_2[1]
        y_24[0] = y_2[2]
        y_21[0] = y_2[0]

        # Computation of total drag of A/C
        s_initial3, s_new3, flag3, bound3 = self.__set_coeff_drag_f3(x_shared)
        # adverse pressure gradient
        g_2[0] = self.base.poly_approx(s_initial3, s_new3, flag3, bound3)
        # Custom Pgrad function: replace by a linear function
        # coeff_dir=(1.04-0.96)/(0.06-0.04)# = 4
        #         g_2[0] = coeff_dir*Z[0]+0.8
        if not true_cstr:
            g_2[0] = g_2[0] - self.PRESSURE_GRADIENT_LIMIT
        return y_2, y_21, y_23, y_24, g_2

    @staticmethod
    def __derive_liftoverdrag(cl_cd, lift_jacobian, drag_jacobian, inv_drag):
        """Compute lift over drag jacobian terms.

        :param cl_cd: lift over drag
        :type cl_cd: numpy array
        :param lift_jacobian: d(lift)/d(var)
        :type lift_jacobian: numpy array
        :param drag_jacobian: d(drag)/d(var)
        :type drag_jacobian: numpy array
        :returns: jacobian of lift over drag
        :rtype: numpy array
        """
        return inv_drag * (lift_jacobian - cl_cd * drag_jacobian)

    @staticmethod
    def __set_coupling_jacobian(jacobian):
        """Set jacobian of coupling variables."""
        jacobian["y_21"]["x_2"] = atleast_2d(jacobian["y_2"]["x_2"][0, :])
        jacobian["y_21"]["x_shared"] = atleast_2d(jacobian["y_2"]["x_shared"][0, :])
        jacobian["y_21"]["y_12"] = atleast_2d(jacobian["y_2"]["y_12"][0, :])
        jacobian["y_21"]["y_32"] = atleast_2d(jacobian["y_2"]["y_32"][0, :])

        jacobian["y_23"]["x_2"] = atleast_2d(jacobian["y_2"]["x_2"][1, :])
        jacobian["y_23"]["x_shared"] = atleast_2d(jacobian["y_2"]["x_shared"][1, :])
        jacobian["y_23"]["y_12"] = atleast_2d(jacobian["y_2"]["y_12"][1, :])
        jacobian["y_23"]["y_32"] = atleast_2d(jacobian["y_2"]["y_32"][1, :])

        jacobian["y_24"]["x_2"] = atleast_2d(jacobian["y_2"]["x_2"][2, :])
        jacobian["y_24"]["x_shared"] = atleast_2d(jacobian["y_2"]["x_shared"][2, :])
        jacobian["y_24"]["y_12"] = atleast_2d(jacobian["y_2"]["y_12"][2, :])
        jacobian["y_24"]["y_32"] = atleast_2d(jacobian["y_2"]["y_32"][2, :])
        return jacobian

    def __initialize_jacobian(self):
        """Initialization of jacobian matrix.

        :returns:  jacobian
        :rtype: dict(dict(ndarray))
        """
        jacobian = {"y_2": {}, "g_2": {}, "y_21": {}, "y_23": {}, "y_24": {}}

        jacobian["y_2"]["x_shared"] = zeros((3, 6), dtype=self.dtype)
        jacobian["y_2"]["x_2"] = zeros((3, 1), dtype=self.dtype)
        jacobian["y_2"]["y_12"] = zeros((3, 2), dtype=self.dtype)
        jacobian["y_2"]["y_32"] = zeros((3, 1), dtype=self.dtype)

        jacobian["g_2"]["x_2"] = zeros((1, 1), dtype=self.dtype)
        jacobian["g_2"]["x_shared"] = zeros((1, 6), dtype=self.dtype)
        jacobian["g_2"]["y_12"] = zeros((1, 2), dtype=self.dtype)
        jacobian["g_2"]["y_32"] = zeros((1, 1), dtype=self.dtype)
        return jacobian

    def derive_blackbox_aerodynamics(self, x_shared, y_12, y_32, x_2):
        """This function calculates drag and lift to drag ratio of A/C.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: numpy array
        :param y_12: shared variables coming from blackbox_structure:

            - y_12[0]: total aircraft weight
            - y_12[1]: wing twist

        :type y_12: numpy array
        :param y_32: shared variables coming from blackbox_propulsion:

            - y_32[0]: engine scale factor

        :type y_32: numpy array
        :param x_2: aero. design variable:

            - x_2[0]: friction coeff

        :type x_2: numpy array
        :returns: jacobian matrix of partial derivatives
        :rtype: dict(dict(ndarray))
        """

        jacobian = self.__initialize_jacobian()
        lift_coeff = self.compute_cl(x_shared, y_12)
        # Modification of drag_coeff_min for ESF and Cf
        s_initial1, s_new1, flag1, bound1 = self.__set_coeff_drag_f1(y_32, x_2)
        fo1, ai_coeff1, aij_coeff1, s_shifted1 = self.base.derive_poly_approx(
            s_initial1, s_new1, flag1, bound1
        )

        drag_coeff_incomp = self.constants[4]
        drag_coeff_min = self.compute_cd_min(x_shared, fo1)

        k_aero = self.compute_k_aero(x_shared)

        # Modification of drag_coeff for wing twist
        s_initial2, s_new2, flag2, bound2 = self.__set_coeff_drag_f2(y_12)
        fo2, ai_coeff2, aij_coeff2, s_shifted2 = self.base.derive_poly_approx(
            s_initial2, s_new2, flag2, bound2
        )

        drag_coeff = self.compute_cd(x_shared, lift_coeff, fo1, fo2)
        cl_cd = lift_coeff / drag_coeff
        cl2 = lift_coeff * lift_coeff
        drag = self.compute_drag(drag_coeff, x_shared)

        rhov2 = self.compute_rhov2(x_shared)
        dyn_pressure = 0.5 * rhov2
        dyn_force = dyn_pressure * x_shared[5]
        inv_drag = 1.0 / drag

        # dLift_dCf
        #        jacobian['y_2']['x_2'][0, 0] = 0.0

        # dDrag_dCf
        dadimcf_dcf = self.compute_dadimcf_dcf(x_2)

        dy2dx2 = dyn_force * fo2 * drag_coeff_incomp * dadimcf_dcf
        dy2dx2 *= (
            ai_coeff1[1]
            + aij_coeff1[1, 0] * s_shifted1[0]
            + aij_coeff1[1, 1] * s_shifted1[1]
        )  # dDrag_dCf
        jacobian["y_2"]["x_2"][1, 0] = dy2dx2
        # d(Lift/Drag)_dCf

        dlod = self.__derive_liftoverdrag(
            cl_cd, jacobian["y_2"]["x_2"][0, 0], jacobian["y_2"]["x_2"][1, 0], inv_drag
        )
        jacobian["y_2"]["x_2"][2, 0] = dlod

        # dLift/dZ= 0.0
        # dDrag/d(t/c)
        dy2dxs = dyn_force * fo2 * 3.05 * 5.0 / 3.0 * x_shared[0] ** (2.0 / 3.0)
        dy2dxs *= (self.math.cos(x_shared[4] * DEG_TO_RAD)) ** 1.5
        jacobian["y_2"]["x_shared"][1, 0] = dy2dxs

        # d(Lift/Drag)/d(t/c)

        # d(Drag)/dh
        drhov2_dh = self.compute_drhov2_dh(x_shared)
        dcl_dh = self.compute_dcl_dh(x_shared, y_12)
        dy2dxs = 0.5 * x_shared[5] * fo2 * drag_coeff_min * drhov2_dh
        add_der = cl2 * drhov2_dh + rhov2 * 2.0 * lift_coeff * dcl_dh
        dy2dxs += 0.5 * x_shared[5] * k_aero * fo2 * add_der
        jacobian["y_2"]["x_shared"][1, 1] = dy2dxs
        # d(Lift/Drag)/dh

        # d(Drag)/dM
        dcd_dmach = self.compute_dcd_dmach(x_shared, y_12, fo2)
        drhov2_dmach = self.compute_drhov2_dmach(x_shared)
        jacobian["y_2"]["x_shared"][1, 2] = (
            0.5 * x_shared[5] * (drag_coeff * drhov2_dmach + rhov2 * dcd_dmach)
        )
        # d(Lift/Drag)/dM

        # d(Drag)/dAR
        #         dk_dAR = self.compute_dk_aero_dAR(x_shared) # = 0.0
        #         jacobian['y_2']['x_shared'][1, 3] = dyn_force * dk_dAR * fo1 * cl2
        # d(Lift/Drag)/dAR

        # d(Drag)/dsweep
        dcd_dsweep = self.compute_dcd_dsweep(x_shared, lift_coeff, fo2)
        jacobian["y_2"]["x_shared"][1, 4] = dyn_force * dcd_dsweep

        # d(Drag)/dsref
        dcd_dsef = self.compute_dcd_dsref(x_shared, y_12, fo2)
        dy2dxs = drag / x_shared[5] + dyn_force * dcd_dsef
        jacobian["y_2"]["x_shared"][1, 5] = dy2dxs

        for i in range(6):
            dy2dxs0i = jacobian["y_2"]["x_shared"][0, i]
            dy2dxs1i = jacobian["y_2"]["x_shared"][1, i]
            dlod = self.__derive_liftoverdrag(cl_cd, dy2dxs0i, dy2dxs1i, inv_drag)
            jacobian["y_2"]["x_shared"][2, i] = dlod

        # dLift/dWt
        jacobian["y_2"]["y_12"][0, 0] = 1.0
        # d(Drag)/dWt
        jacobian["y_2"]["y_12"][1, 0] = 2.0 * y_12[0] * k_aero * fo2 / dyn_force
        # d(Lift/Drag)/dWt
        #         jacobian['y_2']['y_12'][
        #             2, 0] = self.__derive_liftoverdrag(
        #             cl_cd, jacobian['y_2']['y_12'][0, 0],
        #             jacobian['y_2']['x_shared'][1, 0], inv_drag)

        # dLift/dtwist= 0.0
        # d(Drag)/dtwist
        dadimtwist_dadim = self.compute_dadimtwist_dtwist(y_12)
        jacobian["y_2"]["y_12"][1, 1] = (
            dyn_force
            * dadimtwist_dadim
            * (drag_coeff_min + k_aero * lift_coeff * lift_coeff)
            * (ai_coeff2[0] + aij_coeff2[0, 0] * s_shifted2[0])
        )
        # d(Lift/Drag)/dtwist
        #         jacobian['y_2']['y_12'][
        #             2, 1] = inv_drag * (jacobian['y_2']['y_12'][0, 1] -
        #                                 cl_cd * jacobian['y_2']['y_12'][1, 1])
        for i in range(2):
            dy2dy120 = jacobian["y_2"]["y_12"][0, i]
            dy2dy121 = jacobian["y_2"]["y_12"][1, i]
            dy2dy122 = self.__derive_liftoverdrag(cl_cd, dy2dy120, dy2dy121, inv_drag)
            jacobian["y_2"]["y_12"][2, i] = dy2dy122

        # dLift/dESF
        #        jacobian['y_2']['y_32'][0, 0] = 0.0
        # dDrag/dESF
        dadimesf_desf = self.compute_dadimesf_desf(y_32)
        jacobian["y_2"]["y_32"][1, 0] = (
            dyn_force
            * fo2
            * drag_coeff_incomp
            * dadimesf_desf
            * (
                ai_coeff1[0]
                + aij_coeff1[0, 0] * s_shifted1[0]
                + aij_coeff1[0, 1] * s_shifted1[1]
            )
        )
        # d(Lift/Drag)/dtwist
        jacobian["y_2"]["y_32"][2, 0] = self.__derive_liftoverdrag(
            cl_cd,
            jacobian["y_2"]["y_32"][0, 0],
            jacobian["y_2"]["y_32"][1, 0],
            inv_drag,
        )

        # d(dp/dx)/d(t/c)
        s_initial3, s_new3, flag3, bound3 = self.__set_coeff_drag_f3(x_shared)
        _, ai_coeff3, aij_coeff3, s_shifted3 = self.base.derive_poly_approx(
            s_initial3, s_new3, flag3, bound3
        )

        dadimtaper_dtaper = self.compute_dadimtaper_dtaper(x_shared)
        jacobian["g_2"]["x_shared"][0, 0] = dadimtaper_dtaper * (
            ai_coeff3[0] + aij_coeff3[0, 0] * s_shifted3[0]
        )
        #        jacobian['g_2']['x_shared'][0, 1:] = 0.0
        #        jacobian['g_2']['x_2'][0, :] = 0.0
        #        jacobian['g_2']['y_12'][0, :] = 0.0
        #        jacobian['g_2']['y_32'][0, :] = 0.0

        jacobian = self.__set_coupling_jacobian(jacobian)

        return jacobian
