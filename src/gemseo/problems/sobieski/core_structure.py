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
SSBJ Structure computations
***************************
"""
from __future__ import division, unicode_literals

import logging
from math import pi

from numpy import append, array, ones, zeros

LOGGER = logging.getLogger(__name__)
DEG_TO_RAD = pi / 180.0


class SobieskiStructure(object):
    """Class defining structural analysis for Sobieski problem and related method to the
    sructural problem such as disciplines computation, constraints, reference
    optimum."""

    DTYPE_COMPLEX = "complex128"
    DTYPE_DOUBLE = "float64"

    STRESS_LIMIT = 1.09
    TWIST_UPPER_LIMIT = 1.04
    TWIST_LOWER_LIMIT = 0.8

    def __init__(self, sobieski_base):
        """Constructor.

        :param sobieski_base: Sobieski problem
        :type sobieski_base: SobieskiBase
        """
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

    def __set_coeff_twist(self, x_1, y_21, half_span, aero_center):
        """Prepare settings of polynomial function for wing twist computation.

        :param x_1: local design variables
        :type x_1: ndarray
        :param y_21: coupling variables from aerodynamics
        :type y_21: ndarray
        :param half_span: half-span of wing
        :type half_span: ndarray
        :param aero_center: aerodynamic center
        :type aero_center: ndarray
        :returns:  s_initial, s_new, flag, bound
        :rtype: ndarray
        """
        s_initial = array(
            [
                self.x_initial,
                self.half_span_initial,
                self.aero_center_initial,
                self.lift_initial,
            ]
        )
        s_new = array([x_1[1], half_span, aero_center, y_21[0]], dtype=self.dtype)
        flag = array([2, 4, 4, 3], dtype=self.dtype)
        bound = array([0.25, 0.25, 0.25, 0.25], dtype=self.dtype)
        return s_initial, s_new, flag, bound

    def __set_coeff_secthick(self, x_1):
        """Prepare settings of polynomial function for sectional thickness.

        :param x_1: local design variables
        :type x_1: ndarray
        :returns:  s_initial, s_new, flag, bound
        :rtype: ndarray
        """
        s_initial = array([self.x_initial], dtype=self.dtype)
        s_new = array([x_1[1]], dtype=self.dtype)
        flag = array([1], dtype=self.dtype)
        bound = array([0.008], dtype=self.dtype)
        return s_initial, s_new, flag, bound

    def __set_coeff_stress(self, x_1, x_shared, y_21, half_span, aero_center):
        """Prepare settings of polynomial function for stress.

        :param x_1: local design variables
        :type x_1: ndarray
        :param x_shared: system design variables
        :type x_shared: ndarray
        :param y_21: coupling variables from aerodynamics
        :type y_21: ndarray
        :param half_span: half-span of wing
        :type half_span: ndarray
        :param aero_center: aerodynamic center
        :type aero_center: ndarray
        :returns:  s_initial, s_new, flag, bound
        :rtype: ndarray
        """
        s_initial = array(
            [
                self.tc_initial,
                self.lift_initial,
                self.x_initial,
                self.half_span_initial,
                self.aero_center_initial,
            ],
            dtype=self.dtype,
        )
        s_new = array(
            [x_shared[0], y_21[0], x_1[1], half_span, aero_center], dtype=self.dtype
        )
        flag = array([4, 1, 4, 1, 1], dtype=self.dtype)
        return s_initial, s_new, flag

    def __compute_dadimcenter_dcenter(self, x_1):
        """Compute derivative of adim aero center wrt aero center.

        :param x_1: local design variables
        :type x_1: ndarray
        :returns:  derivative of adim aero center wrt aero center
        :rtype: ndarray
        """
        aero_center = self.base.compute_aero_center(x_1)
        return self.base.derive_normalize_s(self.aero_center_initial, aero_center)

    def __compute_dcenter_dlambda(self, x_1):
        """Compute derivative of aero center wrt wing taper ratio.

        :param x_1: local design variables
        :type x_1: ndarray
        :returns:  dR_dlambda
        :rtype: ndarray
        """
        dadimcenter_dcenter = self.__compute_dadimcenter_dcenter(x_1)
        return dadimcenter_dcenter * 1.0 / (3.0 * (1.0 + x_1[0]) ** 2)

    def compute_wing_weight(self, x_shared, x_1, y_21, linearize=False):
        """Compute wing weight.

        :param x_shared: system design variables
        :type x_shared: ndarray
        :param y_21: coupling variables from aerodynamics
        :type y_21: ndarray
        :param x_1: local design variables
        :type x_1: ndarray
        :param linearize: Default value = False)
        :returns: wing weight, wing_weight_coeff, value of poly. function
        :rtype: ndarray
        """
        s_initial, s_new, flag, bound = self.__set_coeff_secthick(x_1)
        if linearize:
            f_o, a_i, a_ij, s_shifted = self.base.derive_poly_approx(
                s_initial, s_new, flag, bound
            )
        else:
            f_o = self.base.poly_approx(s_initial, s_new, flag, bound)
        wing_weight_coeff = (
            0.0051
            * ((y_21[0] * self.constants[2]) ** 0.557)
            * (x_shared[5] ** 0.649)
            * (x_shared[3] ** 0.5)
            * (x_shared[0] ** -0.4)
            * ((1 + x_1[0]) ** 0.1)
            * ((self.math.cos(x_shared[4] * DEG_TO_RAD)) ** -1.0)
            * ((0.1875 * x_shared[5]) ** 0.1)
        )
        wing_weight = wing_weight_coeff * f_o
        if linearize:
            return wing_weight, wing_weight_coeff, f_o, a_i, a_ij, s_shifted
        return wing_weight

    def compute_fuelwing_weight(self, x_shared):
        """Compute fuel wing weight.

        :param x_shared: system design variables
        :type x_shared: ndarray
        :returns: fuel wing weight
        :rtype: ndarray
        """
        thickness = self.base.compute_thickness(x_shared)  # wing thickness
        return (5.0 * x_shared[5] / 18.0) * (2.0 / 3.0 * thickness) * 42.5

    def compute_dfuelwing_dtoverc(self, x_shared):
        """Compute fuel wing weight.

        :param x_shared: system design variables
        :type x_shared: ndarray
        :returns: fuel wing weight
        :rtype: ndarray
        """
        return 212.5 / 27.0 * x_shared[5] ** (3.0 / 2.0) / self.math.sqrt(x_shared[3])

    @staticmethod
    def compute_dfuelwing_dar(x_shared):
        """Compute fuel wing weight.

        :param x_shared: system design variables
        :type x_shared: ndarray
        :returns: fuel wing weight
        :rtype: ndarray
        """
        dfuelwing_dar = 212.5 / 27.0 * x_shared[5] ** (3.0 / 2.0)
        dfuelwing_dar *= x_shared[0] * -0.5 * x_shared[3] ** (-3.0 / 2.0)
        return dfuelwing_dar

    def compute_dfuelwing_dsref(self, x_shared):
        """Compute fuel wing weight.

        :param x_shared: system design variables
        :type x_shared: ndarray
        :returns: fuel wing weight
        :rtype: ndarray
        """
        return (
            637.5
            / 54.0
            * x_shared[5] ** 0.5
            * x_shared[0]
            / self.math.sqrt(x_shared[3])
        )

    def __compute_dhalfspan_dar(self, x_shared):
        """Compute partial derivative of half-span wrt aspect ratio.

        :param x_shared: system design variables
        :type x_shared: ndarray
        :returns:  d(half_span)_d(AR)
        :rtype: ndarray
        """
        dadimspan_dspan = self.__compute_dadimspan_dspan(x_shared)
        return (
            dadimspan_dspan
            * x_shared[5]
            / (8.0 * self.base.compute_half_span(x_shared))
        )

    def __compute_dadimspan_dsref(self, x_shared):
        """Compute partial derivative of half-span wrt wing surface.

        :param x_shared: system design variables
        :type x_shared: ndarray
        :returns:  d(half_span)_d(sref)
        :rtype: ndarray
        """
        dadimspan_dspan = self.__compute_dadimspan_dspan(x_shared)
        return (
            dadimspan_dspan
            * x_shared[3]
            / (8.0 * self.base.compute_half_span(x_shared))
        )

    def __compute_dadimspan_dspan(self, x_shared):
        """Compute partial derivative of adim half-span of polynomial function wrt half-
        span.

        :param x_shared: system design variables
        :type x_shared: ndarray
        :returns:  dadimhalf_span_dhalf_span
        :rtype: ndarray
        """
        half_span = self.base.compute_half_span(x_shared)
        return self.base.derive_normalize_s(self.half_span_initial, half_span)

    def __compute_dadimx_dx(self, x_1):
        """Compute partial derivative of adim sectional area of polynomial function wrt
        sectional area.

        :param x_1: local design variables
        :type x_1: ndarray
        :returns:  dadimx_dx
        :rtype: ndarray
        """
        return self.base.derive_normalize_s(self.x_initial, x_1[1])

    def __compute_dadimtaper_dtaper(self, x_shared):
        """Compute partial derivative of adim taper ratio area of polynomial function
        wrt taper ratio.

        :param x_shared: system design variables
        :type x_shared: ndarray
        :returns:  dadimtaper_dtaper
        :rtype: ndarray
        """
        return self.base.derive_normalize_s(self.tc_initial, x_shared[0])

    def __compute_dadimlift_dlift(self, y_21):
        """Compute partial derivative of adim lift  area of polynomial function wrt
        lift.

        :param y_21: coupling design variables (weight)
        :type y_21: ndarray
        :returns:  dadimtaper_dtaper
        :rtype: ndarray
        """
        return self.base.derive_normalize_s(self.lift_initial, y_21[0])

    @staticmethod
    def compute_weight_ratio(y_1):
        """Computation of weight ratio of Breguet formula.

        :param y_1: shared variables coming from blackbox_structure
            - y_1[0]: total aircraft weight
            - y_1[1]: fuel weight
        :type y_1: ndarray
        :returns: Wt / (Wt -Wf)
        :rtype: ndarray
        """
        return y_1[0] / (y_1[0] - y_1[1])

    def blackbox_structure(self, x_shared, y_21, y_31, x_1, true_cstr=False):
        """This function calculates the weight of the aircraft by structure and adds
        them to obtain a total aircraft weight.

        :param x_shared: shared design variable vector:

            - x_shared[0]: thickness/chord ratio
            - x_shared[1]: altitude
            - x_shared[2]: Mach
            - x_shared[3]: aspect ratio
            - x_shared[4]: wing sweep
            - x_shared[5]: wing surface area

        :type x_shared: ndarray
        :param y_21: lift
        :type y_21: ndarray
        :param y_31: engine weight
        :type y_31: ndarray
        :param x_1: weight design variables:

            - x_1[0]: wing taper ratio
            - x_1[1]: wingbox x-sectional area as poly. funct

        :type x_1: ndarray
        :param true_cstr: Default value = False)
        :returns: g_1,y_1, y_11, y_12, y_14:

            - g_1: vector of constraints for weight analysis

                - g_1[0] to g_1[4]: stress on wing
                - g_1[5]: wing twist as constraint

            - y_1: weight analysis outputs

                - y_1[0]: total aircraft weight
                - y_1[1]: fuel weight
                - y_1[2]: wing twist

            - y_12: shared variables used for aero computations
                (blackbox_aerodynamics)

                - y_12[0]: total aircraft weight
                - y_12[1]: wing twist

            - y_14: shared variables used for range computation
                (BlackBoxMission)

                - y_14[0]: total aircraft weight
                - y_14[1]: fuel weight

        :rtype: ndarray, ndarray, ndarray, ndarray
        """
        y_12 = zeros(2, dtype=self.dtype)
        y_14 = zeros(2, dtype=self.dtype)

        y_1, y_11 = self.poly_structure(x_shared, x_1, y_21, y_31)

        g_1 = self.poly_structure_constraints(x_shared, x_1, y_21)
        g_1[5] = y_1[2]

        # Coupling variables
        y_12[1] = y_1[2]
        y_12[0] = y_1[0]
        y_14[0] = y_1[0]
        y_14[1] = y_1[1]
        if not true_cstr:
            g_1 = append(
                g_1[0:5] - self.STRESS_LIMIT,
                (g_1[5] - self.TWIST_UPPER_LIMIT, self.TWIST_LOWER_LIMIT - g_1[5]),
            )

        return y_1, y_11, y_12, y_14, g_1

    def __initialize_jacobian(self):
        """Initialization of jacobian matrix.

        :returns:  jacobian
        :rtype: dict of dict of ndarray
        """
        # Jacobian matrix as a dictionary
        jacobian = {"y_1": {}, "g_1": {}, "y_12": {}, "y_14": {}, "y_11": {}}

        n_y1 = 3
        jacobian["y_1"]["x_shared"] = zeros((n_y1, 6), dtype=self.dtype)
        jacobian["y_1"]["x_1"] = zeros((n_y1, 2), dtype=self.dtype)
        jacobian["y_1"]["y_21"] = zeros((n_y1, 1), dtype=self.dtype)
        jacobian["y_1"]["y_31"] = zeros((n_y1, 1), dtype=self.dtype)

        n_y11 = 1
        jacobian["y_11"]["x_shared"] = zeros((n_y11, 6), dtype=self.dtype)
        jacobian["y_11"]["x_1"] = zeros((n_y11, 2), dtype=self.dtype)
        jacobian["y_11"]["y_21"] = zeros((n_y11, 1), dtype=self.dtype)
        jacobian["y_11"]["y_31"] = zeros((n_y11, 1), dtype=self.dtype)

        n_y12 = 2
        jacobian["y_12"]["x_shared"] = zeros((n_y12, 6), dtype=self.dtype)
        jacobian["y_12"]["x_1"] = zeros((n_y12, 2), dtype=self.dtype)
        jacobian["y_12"]["y_21"] = zeros((n_y12, 1), dtype=self.dtype)
        jacobian["y_12"]["y_31"] = zeros((n_y12, 1), dtype=self.dtype)

        for key, jac_val in jacobian["y_12"].items():
            jacobian["y_14"][key] = jac_val
        return jacobian

    def derive_blackbox_structure(self, x_shared, y_21, y_31, x_1, true_cstr=False):
        """Compute jacobian matrix of structural analysis y_1 is the vector of
        structural outputs and g_1 are the structural constraints:

        - y_1[0]: total aircraft weight
        - y_1[1]: fuel weight
        - y_1[2]: wing twist

        :param x_shared: shared design variable vector
        :type x_shared: ndarray
        :param y_21: coupling variable from aerodynamics (lift)
        :type y_21: ndarray
        :param y_31: coupling variable from propulsion (Engine weight)
        :type y_31: ndarray
        :param x_1: structure design variable vector

            - x_1[0]: wing taper ratio
            - x_1[1]: wingbox x-sectional area as poly. funct

        :type x_1: ndarray
        :param true_cstr: Default value = False)
        :returns: jacobian : Jacobian matrix
        :rtype: dict(dict(ndarray))
        """
        # Jacobian matrix as a dictionary
        jacobian = self.__initialize_jacobian()

        jacobian = self.derive_poly_structure(jacobian, x_shared, x_1, y_21, y_31)

        # Stress constraints
        jacobian = self.derive_constraints(jacobian, x_shared, x_1, y_21, true_cstr)

        # Twist constraints
        jacobian["g_1"]["x_1"][5, :] = jacobian["y_1"]["x_1"][2, :]
        jacobian["g_1"]["x_shared"][5, :] = jacobian["y_1"]["x_shared"][2, :]
        jacobian["g_1"]["y_21"][5, :] = jacobian["y_1"]["y_21"][2, :]
        jacobian["g_1"]["y_31"][5, :] = jacobian["y_1"]["y_31"][2, :]

        if not true_cstr:
            jacobian["g_1"]["x_1"][6, :] = -jacobian["y_1"]["x_1"][2, :]
            jacobian["g_1"]["x_shared"][6, :] = -jacobian["y_1"]["x_shared"][2, :]
            jacobian["g_1"]["y_21"][6, :] = -jacobian["y_1"]["y_21"][2, :]
            jacobian["g_1"]["y_31"][6, :] = -jacobian["y_1"]["y_31"][2, :]

        # Coupling variables
        jacobian = self.__set_coupling_jacobian(jacobian)

        return jacobian

    @staticmethod
    def __set_coupling_jacobian(jacobian):
        """Set jacobian of coupling variables."""
        # Coupling variables
        for der_v, jac_loc in jacobian["y_1"].items():
            jacobian["y_12"][der_v][1, :] = jac_loc[2, :]
            jacobian["y_12"][der_v][0, :] = jac_loc[0, :]
            jacobian["y_14"][der_v] = jac_loc[0:2, :]
        return jacobian

    def derive_poly_structure(self, jacobian, x_shared, x_1, y_21, y_31):
        """Compute derivatives of structural variables from a polynomial approximation.

        :param jacobian: jacobian matrix
        :type jacobian: dict(dict(ndarray))
        :param x_shared: shared design variable vector
        :type x_shared: ndarray
        :param x_1: structure design variable vector:

                - x_1[0]: wing taper ratio
                - x_1[1]: wingbox x-sectional area as poly. funct

        :type x_1: ndarray
        :param y_21: coupling variable from aerodynamics
        :type y_21: ndarray
        :param y_31: coupling variable from propulsion
        :type y_31: ndarray
        :returns: updated jacobian matrix
        :rtype: dict(dict(ndarray))
        """
        half_span = self.base.compute_half_span(x_shared)  # 1/2 span
        aero_center = self.base.compute_aero_center(x_1)
        # wing aero. center location
        y_1, _ = self.poly_structure(x_shared, x_1, y_21, y_31)

        # Derivation of Fuel Weight Wf=y_1[1]
        # dWf/dx=dWfd(lamda= =0
        # jacobian['y_1']['x_1'][1, :] = 0.0  # Fuel weigh does not depend on x_1

        # dWf/d(t/c)
        jacobian["y_1"]["x_shared"][1, 0] = self.compute_dfuelwing_dtoverc(x_shared)
        #        jacobian['y_1']['x_shared'][1, 1] = 0.0  # dWf/dh
        #        jacobian['y_1']['x_shared'][1, 2] = 0.0  # dWf/dM
        # dWf/d(AR)
        jacobian["y_1"]["x_shared"][1, 3] = self.compute_dfuelwing_dar(x_shared)

        #        jacobian['y_1']['x_shared'][1, 4] = 0.0  # dWf/d(sweep)
        # dWf/dsref
        jacobian["y_1"]["x_shared"][1, 5] = self.compute_dfuelwing_dsref(x_shared)
        # dWf_dLift= 0.0
        # dWf_dWe= 0.0

        # Derivation of wing twist = y_1[2]
        s_initial1, s_1, flag1, bound1 = self.__set_coeff_twist(
            x_1, y_21, half_span, aero_center
        )
        _, a_i, a_ij, s_shifted = self.base.derive_poly_approx(
            s_initial1, s_1, flag1, bound1
        )

        dcenter_dlambda = self.__compute_dcenter_dlambda(x_1)
        jacobian["y_1"]["x_1"][2, 0] = dcenter_dlambda * (
            a_i[2]
            + a_ij[2, 0] * s_shifted[0]
            + a_ij[2, 1] * s_shifted[1]
            + a_ij[2, 2] * s_shifted[2]
            + a_ij[2, 3] * s_shifted[3]
        )

        dadimx_dx = self.__compute_dadimx_dx(x_1)
        jacobian["y_1"]["x_1"][2, 1] = dadimx_dx * (
            a_i[0]
            + a_ij[0, 0] * s_shifted[0]
            + a_ij[0, 1] * s_shifted[1]
            + a_ij[0, 2] * s_shifted[2]
            + a_ij[0, 3] * s_shifted[3]
        )
        #        J['y_1']['x_shared'][2, 0] = 0.0  # dtwist_d(t/c)
        #        J['y_1']['x_shared'][2, 1] = 0.0  # dtwist_dh
        #        J['y_1']['x_shared'][2, 2] = 0.0  # dtwist_dM

        #         dLbar_dL = 1.0 / self.L_initial  # Half-span
        dhalfspan_dar = self.__compute_dhalfspan_dar(x_shared)
        jacobian["y_1"]["x_shared"][2, 3] = dhalfspan_dar * (
            a_i[1]
            + a_ij[1, 0] * s_shifted[0]
            + a_ij[1, 1] * s_shifted[1]
            + a_ij[1, 2] * s_shifted[2]
            + a_ij[1, 3] * s_shifted[3]
        )

        #        jacobian['y_1']['x_shared'][2, 4] = 0.0  # dtwistdsweep

        dhalfspan_dsref = self.__compute_dadimspan_dsref(x_shared)
        jacobian["y_1"]["x_shared"][2, 5] = dhalfspan_dsref * (
            a_i[1]
            + a_ij[1, 0] * s_shifted[0]
            + a_ij[1, 1] * s_shifted[1]
            + a_ij[1, 2] * s_shifted[2]
            + a_ij[1, 3] * s_shifted[3]
        )

        dadimlift_dlift = self.__compute_dadimlift_dlift(y_21)
        jacobian["y_1"]["y_21"][2, 0] = dadimlift_dlift * (
            a_i[3]
            + a_ij[0, 3] * s_shifted[0]
            + a_ij[1, 3] * s_shifted[1]
            + a_ij[2, 3] * s_shifted[2]
        )

        #        jacobian['y_1']['y_31'][2, 0] = 0.0  # dtwist_dWengine

        # Derivation of total weight = y_1[0] (requires derivation of wing weight)
        # Calculation of wingbox X-sectional thickness
        wing_w, ww_coeff, _, a_i, a_ij, s_shifted = self.compute_wing_weight(
            x_shared, x_1, y_21, linearize=True
        )

        # dtotal_weight_d(t/c)
        jacobian["y_1"]["x_shared"][0, 0] = (
            -0.4 * x_shared[0] ** (-1.0) * wing_w + jacobian["y_1"]["x_shared"][1, 0]
        )

        # dtotal_weight_dh = 0.0
        # dtotal_weight_dM = 0.0

        dy11_dz = jacobian["y_1"]["x_shared"]
        # dtotal_weight_dAR
        dy11_dz[0, 3] = 0.5 * x_shared[3] ** (-1.0) * wing_w
        dy11_dz[0, 3] += jacobian["y_1"]["x_shared"][1, 3]

        # d1total_weight_dsweep
        dy11_dz[0, 4] = DEG_TO_RAD
        dy11_dz[0, 4] *= self.math.sin(x_shared[4] * DEG_TO_RAD) * wing_w
        dy11_dz[0, 4] /= self.math.cos(x_shared[4] * DEG_TO_RAD)

        # dtotal_weight_dsref
        dy11_dz[0, 5] = 0.749 * x_shared[5] ** (-1.0) * wing_w
        dy11_dz[0, 5] += jacobian["y_1"]["x_shared"][1, 5]

        # dtotal_weight_d(lambda)
        jacobian["y_1"]["x_1"][0, 0] = (
            0.1 * (1 + x_1[0]) ** (-1.0) * wing_w + jacobian["y_1"]["x_1"][1, 0]
        )

        # dtotal_weight_dx
        jacobian["y_1"]["x_1"][0, 1] = ww_coeff * dadimx_dx
        jacobian["y_1"]["x_1"][0, 1] *= a_i[0] + a_ij[0, 0] * s_shifted[0]

        jacobian["y_1"]["x_1"][0, 1] += jacobian["y_1"]["x_1"][1, 1]

        # dtotal_weight_dLift
        jacobian["y_1"]["y_21"][0, 0] = 0.557 * y_21[0] ** -1 * wing_w
        # dtotal_weight_dWe
        jacobian["y_1"]["y_31"][0, 0] = 1.0

        # y_11 = log(y_1[0])- log((y_1[0] - y_1[1]))
        # dy_11 = dy_1[0]/y_1[0] +(dy_1[1]-dy_1[0])/(y_1[0] - y_1[1])
        #        y_11[3] = y_1[0] / (y_1[0] - y_1[1])
        #         y_11 = array([self.math.log( y_1[0] / (y_1[0] - y_1[1]))])
        #     return arr

        for out_v, jac_y_1 in jacobian["y_1"].items():
            val = jac_y_1[0, :] / y_1[0]
            val -= (jac_y_1[0, :] - jac_y_1[1, :]) / (y_1[0] - y_1[1])

            jacobian["y_11"][out_v][0, :] = val
        return jacobian

    def derive_constraints(self, jacobian, x_shared, x_1, y_21, true_cstr=False):
        """Compute derivative structural constraints from a polynomial approximation.

        :param jacobian: Jacobian matrix
        :type jacobian: dict(dict(ndarray))
        :param x_shared: shared design variable vector
        :type x_shared: ndarray
        :param x_1: structure design variable vector:

            - x_1[0]: wing taper ratio
            - x_1[1]: wingbox x-sectional area as poly. funct

        :type x_1: ndarray
        :param y_21: coupling variable from aerodynamics
        :type y_21: ndarray
        :param true_cstr: Default value = False)
        :returns: updated Jacobian matrix J
        :rtype: dict(dict(ndarray))
        """

        if true_cstr is False:
            n_g = 7
        else:
            n_g = 6
        jacobian["g_1"]["x_shared"] = zeros((n_g, 6), dtype=self.dtype)
        jacobian["g_1"]["x_1"] = zeros((n_g, 2), dtype=self.dtype)
        jacobian["g_1"]["y_21"] = zeros((n_g, 1), dtype=self.dtype)
        jacobian["g_1"]["y_31"] = zeros((n_g, 1), dtype=self.dtype)

        half_span = self.base.compute_half_span(x_shared)  # 1/2 span
        aero_center = self.base.compute_aero_center(x_1)

        # Stress constraints
        s_initial, s_new, flag = self.__set_coeff_stress(
            x_1, x_shared, y_21, half_span, aero_center
        )

        ones_mat = ones(5, dtype=self.dtype)
        for i in range(5):
            bound = 0.1 * ones_mat + i * 0.05 * ones_mat
            _, a_i, a_ij, s_shifted = self.base.derive_poly_approx(
                s_initial, s_new, flag, bound
            )
            dg_dx_1, dg_dz, dg_dy_21, dg_dy_31 = self.__der_constraint(
                x_1, x_shared, y_21, a_i, a_ij, s_shifted
            )
            jacobian["g_1"]["x_1"][i, :] = dg_dx_1[:]
            jacobian["g_1"]["x_shared"][i, :] = dg_dz[:]
            jacobian["g_1"]["y_21"][i, :] = dg_dy_21[:]
            jacobian["g_1"]["y_31"][i, :] = dg_dy_31[:]

        return jacobian

    def poly_structure(self, x_shared, x_1, y_21, y_31):
        """Compute structural variables from a polynomial approximation.

        :param x_shared: shared design variable vector
        :type x_shared: ndarray
        :param x_1: structure design variable vector:

            - x_1[0]: wing taper ratio
            - x_1[1]: wingbox x-sectional area as poly. funct

        :type x_1: ndarray
        :param y_21: coupling variable from aerodynamics
        :type y_21: ndarray
        :param y_31: coupling variable from propulsion: engine weight
        :type y_31: ndarray
        :returns: y_1, g_1: structural variables and structural constraints:

            - y_1[0]: total aircraft weight
            - y_1[1]: fuel weight
            - y_1[2]: wing twist

        :rtype: ndarray, ndarray
        """
        y_1 = zeros(3, dtype=self.dtype)
        #         t = self.base.compute_thickness(x_shared)  # wing thickness

        # Calculation of wing twist
        y_1[2] = self.compute_wing_twist(x_shared, x_1, y_21)

        # Calculation of wingbox X-sectional thickness
        wing_w = self.compute_wing_weight(x_shared, x_1, y_21)
        fuel_wing_weight = self.compute_fuelwing_weight(x_shared)
        y_1_i1 = self.constants[0] + fuel_wing_weight
        y_1[1] = y_1_i1  # Fuel weight
        y_1[0] = self.constants[1] + wing_w + y_1_i1 + y_31[0]

        # This is the mass term in the Breguet range equation.
        y_11 = array([self.math.log(self.compute_weight_ratio(y_1))], dtype=self.dtype)

        return y_1, y_11

    def compute_wing_twist(self, x_shared, x_1, y_21):
        """Compute the wing twist (a.k.a. theta, y_12[1], y_1[2]).

        :param x_shared: shared design variables
        :type x_shared: ndarray
        :param x_1: structure design variable
        :type x_1: ndarray
        :param y_21: coupling variable from aerodynamics
        :type y_21: ndarray
        :returns: the wing twist
        :rtype: float
        """
        # Compute the half span and the aerodynamic center
        half_span = self.base.compute_half_span(x_shared)
        aero_center = self.base.compute_aero_center(x_1)

        s_initial1, s_1, flag1, bound1 = self.__set_coeff_twist(
            x_1, y_21, half_span, aero_center
        )
        return self.base.poly_approx(s_initial1, s_1, flag1, bound1)

    def __der_constraint(self, x_1, x_shared, y_21, a_i, a_ij, s_shifted):
        """Compute derivative of a structural constraints from a polynomial
        approximation on a section.

        :param x_1: structure design variable vector:

            - x_1[0]: wing taper ratio
            - x_1[1]: wingbox x-sectional area as poly. funct

        :type x_1: ndarray
        :param x_shared: shared design variable vector
        :type x_shared: ndarray
        :param a_i: linear polynomial coeff
        :type a_i: ndarray
        :param a_ij: quadratic polynomial coeff
        :type a_ij: ndarray
        :param s_shifted: normalized design variables
            (local at polynomial function)
        :type s_shifted: ndarray
        :returns: update Jacobian matrix
        :rtype: dict(dict(ndarrays))
        """
        dcenter_dlambda = self.__compute_dcenter_dlambda(x_1)

        dadimx_dx = self.__compute_dadimx_dx(x_1)

        dadimhalfspan_dar = self.__compute_dhalfspan_dar(x_shared)

        dadimhalfspan_dsref = self.__compute_dadimspan_dsref(x_shared)

        dadimtaper_dtaper = self.__compute_dadimtaper_dtaper(x_shared)

        dadim_lift_dlift = self.__compute_dadimlift_dlift(y_21)

        dg_dz = zeros((1, 6), dtype=self.dtype)
        dg_dx_1 = zeros((1, 2), dtype=self.dtype)
        dg_dy_21 = zeros((1, 1), dtype=self.dtype)
        dg_dy_31 = zeros((1, 1), dtype=self.dtype)
        dg_dx_1[0, 0] = dcenter_dlambda * (
            a_i[4]
            + a_ij[4, 0] * s_shifted[0]
            + a_ij[4, 1] * s_shifted[1]
            + a_ij[4, 2] * s_shifted[2]
            + a_ij[4, 3] * s_shifted[3]
            + a_ij[4, 4] * s_shifted[4]
        )

        # dsigma/dx
        dg_dx_1[0, 1] = dadimx_dx * (
            a_i[2]
            + a_ij[2, 0] * s_shifted[0]
            + a_ij[2, 1] * s_shifted[1]
            + a_ij[2, 2] * s_shifted[2]
            + a_ij[2, 3] * s_shifted[3]
            + a_ij[2, 4] * s_shifted[4]
        )

        # dsigma1/d(t/c)
        dg_dz[0, 0] = dadimtaper_dtaper * (
            a_i[0]
            + a_ij[0, 0] * s_shifted[0]
            + a_ij[0, 1] * s_shifted[1]
            + a_ij[0, 2] * s_shifted[2]
            + a_ij[0, 3] * s_shifted[3]
            + a_ij[0, 4] * s_shifted[4]
        )

        # dsigma1/dh
        dg_dz[0, 1] = 0.0

        # dsigma1/dM
        dg_dz[0, 2] = 0.0

        # dsigma1/dAR
        dg_dz[0, 3] = dadimhalfspan_dar * (
            a_i[3]
            + a_ij[3, 0] * s_shifted[0]
            + a_ij[3, 1] * s_shifted[1]
            + a_ij[3, 2] * s_shifted[2]
            + a_ij[3, 3] * s_shifted[3]
            + a_ij[3, 4] * s_shifted[4]
        )

        # dsigma1/dsweep
        dg_dz[0, 4] = 0.0

        # dsigma1/dsref
        dg_dz[0, 5] = dadimhalfspan_dsref * (
            a_i[3]
            + a_ij[3, 0] * s_shifted[0]
            + a_ij[3, 1] * s_shifted[1]
            + a_ij[3, 2] * s_shifted[2]
            + a_ij[3, 3] * s_shifted[3]
            + a_ij[3, 4] * s_shifted[4]
        )

        # dsigma1/dLift
        dg_dy_21[0, 0] = dadim_lift_dlift * (
            a_i[1]
            + a_ij[1, 0] * s_shifted[0]
            + a_ij[1, 1] * s_shifted[1]
            + a_ij[1, 2] * s_shifted[2]
            + a_ij[1, 3] * s_shifted[3]
            + a_ij[1, 4] * s_shifted[4]
        )

        # dsigma1/dWengine
        dg_dy_31[0, 0] = 0.0

        return dg_dx_1, dg_dz, dg_dy_21, dg_dy_31

    def poly_structure_constraints(self, x_shared, x_1, y_21):
        """Compute structural constraints from a polynomial approximation.

        :param x_shared: shared design variable vector
        :type x_shared: ndarray
        :param x_1: structure design variable vector:

            - x_1[0]: wing taper ratio
            - x_1[1]: wingbox x-sectional area as poly. funct

        :type x_1: ndarray
        :param y_21: coupling variable from aerodynamics
        :type y_21: ndarray
        :returns: g_1: structural constraints
        :rtype: ndarray
        """

        half_span = self.base.compute_half_span(x_shared)  # 1/2 span
        # wing aero. center location
        aero_center = self.base.compute_aero_center(x_1)

        g_1 = zeros(6, dtype=self.dtype)
        s_initial, s_new, flag = self.__set_coeff_stress(
            x_1, x_shared, y_21, half_span, aero_center
        )
        loc_ones = ones(5, dtype=self.dtype)
        for i in range(5):
            bound = 0.1 * loc_ones + i * 0.05 * loc_ones
            g_1[i] = self.base.poly_approx(s_initial, s_new, flag, bound)

        return g_1
