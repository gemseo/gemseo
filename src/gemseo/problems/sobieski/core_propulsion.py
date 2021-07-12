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
SSBJ Propulsion computations
****************************
"""
from __future__ import division, unicode_literals

import logging
from math import pi

from numpy import append, array, atleast_2d, zeros

LOGGER = logging.getLogger(__name__)
DEG_TO_RAD = pi / 180.0


class SobieskiPropulsion(object):
    """Class defining propulsion analysis for Sobieski problem and related method to the
    propulsion problem such as disciplines computation, constraints, reference
    optimum."""

    DTYPE_COMPLEX = "complex128"
    DTYPE_DOUBLE = "float64"

    ESF_UPPER_LIMIT = 1.5
    ESF_LOWER_LIMIT = 0.5
    TEMPERATURE_LIMIT = 1.02

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
        # Surface fit to engine deck with least square method
        # Polynomial coefficients for SFC computation
        self.sfc_coeff = array(
            [
                1.13238425638512,
                1.53436586044561,
                -0.00003295564466,
                -0.00016378694115,
                -0.31623315541888,
                0.00000410691343,
                -0.00005248000590,
                -0.00000000008574,
                0.00000000190214,
                0.00000001059951,
            ],
            dtype=self.dtype,
        )
        # Polynomial coefficients for throttle constraint
        self.thua_coeff = array(
            [
                11483.7822254806,
                10856.2163466548,
                -0.5080237941,
                3200.157926969,
                -0.1466251679,
                0.0000068572,
            ],
            dtype=self.dtype,
        )
        self.throttle_coeff = self.dtype(16168.6)

    def __set_coeff_temp(self, x_shared, x_3):
        """Prepare settings of polynomial function for temperature.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :param x_3: local design variables
        :type x_3: numpy array
        :returns:  s_initial, s_new, flag, bound
        :rtype: numpy array
        """
        s_initial = array(
            [self.mach_initial, self.h_initial, self.throttle_initial], dtype=self.dtype
        )
        s_new = array([x_shared[2], x_shared[1], x_3[0]], dtype=self.dtype)
        flag = array([2, 4, 2], dtype=self.dtype)
        bound = array([0.25, 0.25, 0.25], dtype=self.dtype)
        return s_initial, s_new, flag, bound

    def compute_dim_throttle(self, adim_throttle):
        """Compute a dimensioned value of throttle from adim value.

        :param adim_throttle: local design vector
        :type adim_throttle: numpy array
        :returns: throttle
        :rtype: numpy array
        """
        return adim_throttle * self.throttle_coeff

    def compute_sfc(self, x_shared, adim_throttle):
        """Compute Specific Fuel Consumption (SFC) from global design variables (M, h,)
        and local design variable (throttle) by polynomial function.

        :param x_shared: global design vector
        :type x_shared: numpy array
        :param adim_throttle: local design vector
        :type adim_throttle: numpy array
        :returns: sfc
        :rtype: numpy array
        """
        throttle = self.compute_dim_throttle(adim_throttle)
        sfc = (
            self.sfc_coeff[0]
            + self.sfc_coeff[1] * x_shared[2]
            + self.sfc_coeff[2] * x_shared[1]
            + self.sfc_coeff[3] * throttle
            + self.sfc_coeff[4] * x_shared[2] ** 2
            + 2 * x_shared[1] * x_shared[2] * self.sfc_coeff[5]
            + 2 * throttle * x_shared[2] * self.sfc_coeff[6]
            + self.sfc_coeff[7] * x_shared[1] ** 2
            + 2 * throttle * x_shared[1] * self.sfc_coeff[8]
            + self.sfc_coeff[9] * throttle ** 2
        )
        return sfc

    def compute_throttle_ua(self, x_shared):
        """Compute throttle upper limit from global design variables (M, h,)

        :param x_shared: global design vector
        :type x_shared: numpy array
        :returns: throttle_uA
        :rtype: numpy array
        """
        throttle_ua = (
            self.thua_coeff[0]
            + self.thua_coeff[1] * x_shared[2]
            + self.thua_coeff[2] * x_shared[1]
            + self.thua_coeff[3] * x_shared[2] * x_shared[2]
            + 2 * self.thua_coeff[4] * x_shared[2] * x_shared[1]
            + self.thua_coeff[5] * x_shared[1] * x_shared[1]
        )
        return throttle_ua

    def compute_throttle_constraint(self, x_shared, adim_throttle):
        """Compute throttle constraint (M, h,) and local design variable (throttle) by
        polynomial function.

        :param x_shared: global design vector
        :type x_shared: numpy array
        :param adim_throttle: local design vector
        :type adim_throttle: numpy array
        :returns: throttle constraint as throttle / throttle_ua - 1.0
        :rtype: numpy array
        """
        throttle = self.compute_dim_throttle(adim_throttle)
        throttle_ua = self.compute_throttle_ua(x_shared)
        throttle_constraint = throttle / throttle_ua - 1.0  # throttle setting
        return throttle_constraint

    def compute_dthrconst_dthrottle(self, x_shared):
        """Compute derivative of throttle constraint wrt throttle.

        :param x_shared: global design vector
        :type x_shared: numpy array
        :returns: dthrottle_constraint_dthrottle
        :rtype: numpy array
        """
        throttle_ua = self.compute_throttle_ua(x_shared)
        return self.throttle_coeff / throttle_ua

    def compute_dthrcons_dh(self, x_shared, adim_throttle):
        """Compute derivative of throttle constraint wrt altitude.

        :param x_shared: global design vector
        :type x_shared: numpy array
        :param adim_throttle: local design vector
        :type adim_throttle: numpy array
        :returns: dthrottle_constraint_dh
        :rtype: numpy array
        """
        throttle = self.compute_dim_throttle(adim_throttle)
        throttle_ua = self.compute_throttle_ua(x_shared)
        dthrottle_ua_dh = (
            self.thua_coeff[2]
            + 2 * self.thua_coeff[4] * x_shared[2]
            + 2.0 * self.thua_coeff[5] * x_shared[1]
        )
        return -throttle * dthrottle_ua_dh / (throttle_ua * throttle_ua)

    def compute_dthrconst_dmach(self, x_shared, adim_throttle):
        """Compute derivative of throttle constraint wrt Mach number.

        :param x_shared: global design vector
        :type x_shared: numpy array
        :param adim_throttle: local design vector
        :type adim_throttle: numpy array
        :returns: dthrottle_constraint_dmach
        :rtype: numpy array
        """
        throttle = self.compute_dim_throttle(adim_throttle)
        throttle_ua = self.compute_throttle_ua(x_shared)
        dthrottle_ua_dmach = (
            self.thua_coeff[1]
            + 2 * self.thua_coeff[3] * x_shared[2]
            + 2.0 * self.thua_coeff[4] * x_shared[1]
        )
        return -throttle * dthrottle_ua_dmach / (throttle_ua * throttle_ua)

    def compute_esf(self, drag, adim_throttle):
        """Compute engine scale factor.

        :param drag: drag
        :type drag: float
        :param adim_throttle: throttle (not dimensioned)
        :type adim_throttle: float
        :returns: Engine Scale Factor
        :rtype: numpy array
        """
        return drag / (3.0 * self.compute_dim_throttle(adim_throttle))

    def compute_desf_ddrag(self, adim_throttle):
        """Compute derivative of ESF wrt aero drag.

        :param adim_throttle: local design variables
        :type adim_throttle: numpy array
        :returns: derivative of ESF wrt drag
        :rtype: numpy array
        """
        return 1.0 / (3 * self.compute_dim_throttle(adim_throttle))

    def compute_desf_dthrottle(self, drag, adim_throttle):
        """Compute derivative of ESF wrt aero drag.

        :param drag: coupling design variables
        :type drag: numpy array
        :param adim_throttle: local design variables
        :type adim_throttle: numpy array
        :returns: derivative of ESF wrt drag
        :rtype: numpy array
        """
        throttle = self.compute_dim_throttle(adim_throttle)
        return -self.throttle_coeff * drag / (3.0 * throttle ** 2)

    def compute_temp(self, x_shared, x_3):
        """Compute engine temperature.

        :param x_3: local design vector
        :type x_3: numpy array
        :param x_shared: global design vector
        :type x_shared: numpy array
        :returns: engine temperature
        :rtype: numpy array
        """
        s_initial, s_new, flag, bound = self.__set_coeff_temp(x_shared, x_3)
        return self.base.poly_approx(s_initial, s_new, flag, bound)

    def compute_engine_weight(self, esf):
        """Compute engine weight.

        :param esf: engine scale factor
        :returns: engine weight
        :rtype: numpy array
        """
        return self.constants[3] * (esf ** 1.05) * 3

    def blackbox_propulsion(self, x_shared, y_23, x_3, true_cstr=False):
        """This function calculates fuel comsumption, engine weight and engine scale
        factor.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: numpy array
        :param y_23: shared variables coming from blackbox_aerodynamics (drag)
        :type y_23: numpy array
        :param x_3: power/propulsion design variable (throttle setting)
        :type x_3: numpy array
        :param true_cstr: Default value = False)
        :returns: y_3, y_34, y_31, y_32, g_3:

            - y_3: output variables for propulsion analysis

                - y_3[0]: SFC
                - y_3[1]: engine weight
                - y_3[2]: engine scale factor

            - y_34: shared variable for BlackBoxMission (SFC)
            - y_31: shared variable for blackbox_structure (engine weight)
            - y_32: shared variable for blackbox_aerodynamics (esf)
            - g_3: propulsion constraints

                - g_3[0]: engine scale factor constraint
                - g_3[1]: engine temperature
                - g_3[2]: throttle setting constraint

        :rtype: numpy array, numpy array, numpy array, numpy array, numpy array
        """
        y_3 = zeros(3, dtype=self.dtype)
        g_3 = zeros(3, dtype=self.dtype)
        y_31 = zeros(1, dtype=self.dtype)
        y_32 = zeros(1, dtype=self.dtype)
        y_34 = zeros(1, dtype=self.dtype)

        esf = self.compute_esf(y_23[0], x_3[0])
        y_3[2] = esf
        y_3[1] = self.compute_engine_weight(esf)
        y_3[0] = self.compute_sfc(x_shared, x_3[0])

        y_31[0] = y_3[1]
        y_34[0] = y_3[0]
        y_32[0] = y_3[2]

        # THIS SECTION COMPUTES SFC, esf, AND ENGINE WEIGHT
        g_3[0] = y_3[2]  # engine scale factor

        # engine temperature
        temp = self.compute_temp(x_shared, x_3)
        g_3[1] = temp

        g_3[2] = self.compute_throttle_constraint(x_shared, x_3[0])

        if not true_cstr:
            g_3 = append(
                g_3[0] - self.ESF_UPPER_LIMIT,
                (
                    self.ESF_LOWER_LIMIT - g_3[0],
                    g_3[2],
                    g_3[1] - self.TEMPERATURE_LIMIT,
                ),
            )
        return y_3, y_34, y_31, y_32, g_3

    def compute_dengineweight_dvar(self, esf, desf_dx):
        """Computes derivative of engine weight wrt to a variable(drag or throttle)

        :param esf: ESF
        :type esf: float
        :param desf_dx: partial derivative of ESF wrt to input variable
        :type desf_dx: numpy array
        :returns: derivative of engine weight wrt a variable (drag or throttle)
        :rtype: numpy array
        """
        return 3 * self.constants[3] * 1.05 * desf_dx * esf ** 0.05

    def compute_dsfc_dthrottle(self, x_shared, adim_throttle):
        """Compute derivative of sfc constraint wrt throttle.

        :param x_shared: global design vector
        :type x_shared: numpy array
        :param adim_throttle: local design vector
        :type adim_throttle: numpy array
        :returns: dsfc_dthrottle
        :rtype: numpy array
        """
        throttle = self.compute_dim_throttle(adim_throttle)
        dsfc_dthrottle = (
            self.sfc_coeff[3]
            + 2 * x_shared[2] * self.sfc_coeff[6]
            + 2 * x_shared[1] * self.sfc_coeff[8]
            + 2 * self.sfc_coeff[9] * throttle
        )
        dsfc_dthrottle *= self.throttle_coeff
        return dsfc_dthrottle

    def compute_dsfc_dh(self, x_shared, adim_throttle):
        """Compute derivative of sfc constraint wrt altitude.

        :param x_shared: global design vector
        :type x_shared: numpy array
        :param adim_throttle: local design vector
        :type adim_throttle: numpy array
        :returns: dsfc_dh
        :rtype: numpy array
        """
        throttle = self.compute_dim_throttle(adim_throttle)
        dsfc_dh = (
            self.sfc_coeff[2]
            + 2 * x_shared[2] * self.sfc_coeff[5]
            + +2 * self.sfc_coeff[7] * x_shared[1]
            + 2 * throttle * self.sfc_coeff[8]
        )
        return dsfc_dh

    def compute_dsfc_dmach(self, x_shared, adim_throttle):
        """Compute derivative of sfc constraint wrt Mach number.

        :param x_shared: global design vector
        :type x_shared: numpy array
        :param adim_throttle: local design vector
        :type adim_throttle: numpy array
        :returns: dsfc_dmach
        :rtype: numpy array
        """
        throttle = self.compute_dim_throttle(adim_throttle)
        dsfc_dmach = (
            self.sfc_coeff[1]
            + 2 * self.sfc_coeff[4] * x_shared[2]
            + 2 * x_shared[1] * self.sfc_coeff[5]
            + 2 * throttle * self.sfc_coeff[6]
        )
        return dsfc_dmach

    def __dadimthrottle_dthrottle(self, x_3):
        """Compute partial derivative of adim throttle of polynomial function wrt
        throttle.

        :param x_3: local design variables
        :type x_3: numpy array
        :returns:  dadimthrottle_dthrottle
        :rtype: numpy array
        """
        return self.base.derive_normalize_s(self.throttle_initial, x_3[0])

    def __compute_dadimh_dh(self, x_shared):
        """Compute partial derivative of adim throttle of polynomial function wrt
        altitude.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns:  dadimh_dh
        :rtype: numpy array
        """
        return self.base.derive_normalize_s(self.h_initial, x_shared[1])

    def __compute_dadimmach_dmach(self, x_shared):
        """Compute partial derivative of adim throttle of polynomial function wrt Mach
        number.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns:  dadimthrottle_dthrottle
        :rtype: numpy array
        """
        return self.base.derive_normalize_s(self.mach_initial, x_shared[2])

    def __initialize_jacobian(self, true_cstr):
        """Initialization of jacobian matrix.

        :param true_cstr:
        :type true_cstr: logical
        :returns:  jacobian
        :rtype: dict of dict of numpy array
        """
        # Jacobian matrix as a dictionary
        jacobian = {"y_3": {}, "g_3": {}, "y_31": {}, "y_32": {}, "y_34": {}}

        jacobian["y_3"]["x_3"] = zeros((3, 1), dtype=self.dtype)
        jacobian["y_3"]["x_shared"] = zeros((3, 6), dtype=self.dtype)
        jacobian["y_3"]["y_23"] = zeros((3, 1), dtype=self.dtype)
        if not true_cstr:
            n_constraints = 4
        else:
            n_constraints = 3
        jacobian["g_3"]["x_3"] = zeros((n_constraints, 1), dtype=self.dtype)
        jacobian["g_3"]["x_shared"] = zeros((n_constraints, 6), dtype=self.dtype)
        jacobian["g_3"]["y_23"] = zeros((n_constraints, 1), dtype=self.dtype)

        return jacobian

    def derive_blackbox_propulsion(self, x_shared, y_23, x_3, true_cstr=False):
        """Compute jacobian matrix of propulsion analysis.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: numpy array
        :param y_23: shared variables coming from blackbox_aerodynamics (drag)
        :type y_23: numpy array
        :param x_3: power/propulsion design variable (throttle setting)
        :type x_3: numpy array
        :param true_cstr: Default value = False)
        :returns: jacobian : Jacobian matrix
        :rtype: dict(dict(ndarray))
        """
        # Jacobian matrix as a dictionary
        jacobian = self.__initialize_jacobian(true_cstr)

        dg_3_dx_3 = zeros((3, 1), dtype=self.dtype)
        dg_3_dxs = zeros((3, 6), dtype=self.dtype)
        dg_3_dy_23 = zeros((3, 1), dtype=self.dtype)

        drag = y_23[0]

        esf = self.compute_esf(y_23[0], x_3[0])

        # dSFC_dthrottle
        jacobian["y_3"]["x_3"][0, 0] = self.compute_dsfc_dthrottle(x_shared, x_3[0])

        # dESF_dthrottle
        jacobian["y_3"]["x_3"][2, 0] = self.compute_desf_dthrottle(drag, x_3[0])
        # dengineweight_dthrottle
        jacobian["y_3"]["x_3"][1, :] = self.compute_dengineweight_dvar(
            esf, jacobian["y_3"]["x_3"][2, 0]
        )
        # dSFC_d(t/c) = 0
        # dESF_d(t/c) = 0
        # dengineweight_d(t/c) = 0

        # dSFC_dh
        jacobian["y_3"]["x_shared"][0, 1] = self.compute_dsfc_dh(x_shared, x_3[0])
        # dESF_dh= 0.0
        # dengineweight_dh= 0.0

        # dSFC_dM
        jacobian["y_3"]["x_shared"][0, 2] = self.compute_dsfc_dmach(x_shared, x_3[0])
        # dESF_dM= 0.0
        # dengineweight_dM= 0.0

        #        jacobian['y_3']['x_shared'][:, 2:] = 0.0

        # dSFC_ddrag
        #        jacobian['y_3']['y_23'][0, 0] = 0.0
        # dESF_ddrag
        jacobian["y_3"]["y_23"][2, 0] = self.compute_desf_ddrag(x_3[0])
        # dengineweight_ddrag
        jacobian["y_3"]["y_23"][1, :] = self.compute_dengineweight_dvar(
            esf, jacobian["y_3"]["y_23"][2, 0]
        )

        # dtemp_ddrag
        #        jacobian['g_3']['y_23'][0, 0] = 0.0
        s_initial, s_new, flag, bound = self.__set_coeff_temp(x_shared, x_3)
        _, ai_coeff, aij_coeff, s_shifted = self.base.derive_poly_approx(
            s_initial, s_new, flag, bound
        )

        #        dadimtemp_dtemp = self.__compute_dadimtemp_dtemp(x_shared)

        # g_3[0, :] = ESF
        dg_3_dx_3[0, :] = jacobian["y_3"]["x_3"][2, :]
        # dtemp_dthrottle
        dg_3_dx_3[1, 0] = self.__dadimthrottle_dthrottle(x_3) * (
            ai_coeff[2]
            + aij_coeff[2, 0] * s_shifted[0]
            + aij_coeff[2, 1] * s_shifted[1]
            + aij_coeff[2, 2] * s_shifted[2]
        )
        # d(throttle-throttle_ua)_dthrottle
        dg_3_dx_3[2, 0] = self.compute_dthrconst_dthrottle(x_shared)

        # g_3[0, :] = ESF
        dg_3_dxs[0, :] = jacobian["y_3"]["x_shared"][2, :]

        # dtemp_d(t/c)= 0.0
        # d(throttle-throttle_ua)_d(t/c)= 0.0

        # dtemp_dh
        dg_3_dxs[1, 1] = self.__compute_dadimh_dh(x_shared) * (
            ai_coeff[1]
            + aij_coeff[1, 0] * s_shifted[0]
            + aij_coeff[1, 1] * s_shifted[1]
            + aij_coeff[1, 2] * s_shifted[2]
        )
        # d(throttle-throttle_ua)_dh
        dg_3_dxs[2, 1] = self.compute_dthrcons_dh(x_shared, x_3[0])

        # dtemp_dM
        dg_3_dxs[1, 2] = self.__compute_dadimmach_dmach(x_shared) * (
            ai_coeff[0]
            + aij_coeff[0, 0] * s_shifted[0]
            + aij_coeff[0, 1] * s_shifted[1]
            + aij_coeff[0, 2] * s_shifted[2]
        )
        # d(throttle-throttle_ua)_dM
        dg_3_dxs[2, 2] = self.compute_dthrconst_dmach(x_shared, x_3[0])

        jacobian = self.__set_coupling_jacobian(jacobian)

        # THIS SECTION COMPUTES SFC, ESF, AND ENGINE WEIGHT
        dg_3_dxs[0, :] = jacobian["y_3"]["x_shared"][2, :]
        dg_3_dy_23[0, :] = jacobian["y_3"]["y_23"][2, :]

        if true_cstr:
            jacobian["g_3"]["x_3"] = dg_3_dx_3
            jacobian["g_3"]["x_shared"] = dg_3_dxs
            jacobian["g_3"]["y_23"] = dg_3_dy_23
        else:
            for i_g_jac, i_orig in enumerate([0, 0, 2, 1]):
                jacobian["g_3"]["x_shared"][i_g_jac, :] = dg_3_dxs[i_orig, :]
                jacobian["g_3"]["y_23"][i_g_jac, :] = dg_3_dy_23[i_orig, :]
                jacobian["g_3"]["x_3"][i_g_jac, :] = dg_3_dx_3[i_orig, :]
                if i_g_jac == 1:
                    jacobian["g_3"]["x_shared"][i_g_jac, :] *= -1
                    jacobian["g_3"]["y_23"][i_g_jac, :] *= -1
                    jacobian["g_3"]["x_3"][i_g_jac, :] *= -1
        return jacobian

    @staticmethod
    def __set_coupling_jacobian(jacobian):
        """Set jacobian of coupling variables."""
        jacobian["y_31"]["x_3"] = atleast_2d(jacobian["y_3"]["x_3"][1, :])
        jacobian["y_31"]["x_shared"] = atleast_2d(jacobian["y_3"]["x_shared"][1, :])
        jacobian["y_31"]["y_23"] = atleast_2d(jacobian["y_3"]["y_23"][1, :])

        jacobian["y_32"]["x_3"] = atleast_2d(jacobian["y_3"]["x_3"][2, :])
        jacobian["y_32"]["x_shared"] = atleast_2d(jacobian["y_3"]["x_shared"][2, :])
        jacobian["y_32"]["y_23"] = atleast_2d(jacobian["y_3"]["y_23"][2, :])

        jacobian["y_34"]["x_3"] = atleast_2d(jacobian["y_3"]["x_3"][0, :])
        jacobian["y_34"]["x_shared"] = atleast_2d(jacobian["y_3"]["x_shared"][0, :])
        jacobian["y_34"]["y_23"] = atleast_2d(jacobian["y_3"]["y_23"][0, :])
        return jacobian
