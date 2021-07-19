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
SSBJ base class
***************
"""
from __future__ import division, unicode_literals

import cmath
import logging
import math

from numpy import array, asarray, complex128, concatenate, dot, float64, zeros
from numpy.linalg import lstsq
from scipy import linalg
from six import string_types

LOGGER = logging.getLogger(__name__)


DEG_TO_RAD = math.pi / 180.0


class SobieskiBase(object):
    """Class defining Sobieski problem and related method to the problem such as
    disciplines computation, constraints, reference optimum."""

    DTYPE_COMPLEX = "complex128"
    DTYPE_DOUBLE = "float64"
    DTYPE_DEFAULT = DTYPE_COMPLEX

    def __init__(self, dtype):
        """Constructor.

        :param dtype: data type
        :type dtype: str
        """
        self.dtype = dtype
        if dtype == complex128:
            self.math = cmath
        elif dtype == float64:
            self.math = math
        elif dtype == self.DTYPE_DOUBLE:
            self.math = math
            self.dtype = float64
        elif dtype == self.DTYPE_COMPLEX:
            self.math = cmath
            self.dtype = complex128
        else:
            raise ValueError("Unknown dtype : " + str(dtype))

        self.i_0 = None

    def get_default_x0(self):
        """Return a default initial value for design variables.

        :returns: initial design variables
        :rtype: numpy array

        warning:
        ##DO NOT CHANGE VALUE: THEY ARE USED FOR POLYNOMIAL APPROXIMATION ##
        """
        return array(
            [0.25, 1.0, 1.0, 0.5, 0.05, 45000.0, 1.6, 5.5, 55.0, 1000.0],
            dtype=self.dtype,
        )

    def default_constants(self):
        """Definition of constants vector constants for Sobieski problem.

        :returns: constant vector
        :rtype: numpy array
        """
        constants = zeros(5, dtype=self.dtype)

        # Constants of problem
        constants[0] = 2000.0  # minimum fuel weight
        constants[1] = 25000.0  # miscellaneous weight
        constants[2] = 6.0  # Maximum load factor
        constants[3] = 4360.0  # Engine weight reference
        constants[4] = 0.01375  # Minimum drag coefficient
        return constants

    @staticmethod
    def get_sobieski_bounds_tuple():
        """Set the input design bounds and return them as a tuple of tuples.

        :returns: Subtuple is build with
            lower bound and upper bound.
        :rtype: tuple(tuple)
        """
        bounds_tuple = (
            (0.1, 0.4),
            (0.75, 1.25),
            (0.75, 1.25),
            (0.1, 1),
            (0.01, 0.09),
            (30000.0, 60000.0),
            # (0.01, 0.09) with threshold, (0.038, 0.06)
            # without threshold
            (1.4, 1.8),
            (2.5, 8.5),
            (40.0, 70.0),
            (500.0, 1500.0),
        )

        return bounds_tuple

    @classmethod
    def get_sobieski_bounds(cls):
        """Set the input design bounds and return them as 2 numpy arrays.

        :returns: upper and lower bounds
        :rtype: numpy array, numpy array
        """
        bounds_tuple = cls.get_sobieski_bounds_tuple()
        bounds_array = asarray(bounds_tuple)
        upper_bounds = bounds_array[:, 1]
        lower_bounds = bounds_array[:, 0]
        return upper_bounds, lower_bounds

    @classmethod
    def get_bounds_by_name(cls, variables_names):
        """Return bounds of design variables and coupling variables.

        :param variables_names: names of variables
        :type variables_names: str or list(str)
        :returns: lower bound and upper bound
        """

        if isinstance(variables_names, string_types):
            variables_names = [variables_names]
        bounds_tuple = cls.get_sobieski_bounds_tuple()
        bounds_dict = {
            "x_1": array(bounds_tuple[0:2]),
            "x_2": array([bounds_tuple[2]]),
            "x_3": array([bounds_tuple[3]]),
            "x_shared": array(bounds_tuple[4:10]),
            "y_14": array([(4.97e04, 5.14e04), (-1.54e04, 3.0e04)]),
            "y_12": array([(4.97e04, 5.15e04), (0.9, 1.00)]),
            "y_21": array([(4.97e04, 5.15e04)]),
            "y_23": array([(6.73e03, 1.76e04)]),
            "y_24": array([(0.88, 7.42)]),
            "y_31": array([(5.92e03, 6.79e03)]),
            "y_32": array([(0.47, 0.53)]),
            "y_34": array([(0.88, 1.32)]),
        }

        for yuk in ["y_21", "y_14", "y_12", "y_23", "y_24", "y_31", "y_32", "y_34"]:
            bounds_dict[yuk][:, 0] = bounds_dict[yuk][:, 0] * 0.5
            bounds_dict[yuk][:, 1] = bounds_dict[yuk][:, 1] * 1.5

        bounds = bounds_dict[variables_names[0]]
        #         LOGGER.debug(15 * " " + "get_bounds_by_name")
        for var_name in variables_names[1:]:
            #             LOGGER.debug("var_name " + var_name)
            bounds = concatenate((bounds, bounds_dict[var_name]), axis=0)
        return bounds[:, 0], bounds[:, 1]

    def __compute_mtx_shifted(self, s_bound, index):
        """Compute a matrix of shifted values of design variables.

        :param s_bound: vector of bounds used to control slope of
                polynomial function
        :type s_bound: numpy array
        :param index: index of design variable in polynomial function
        :type index: integer
        :returns: mtx_shifted
        :rtype: numpy array
        """
        #         s_mid = 0.0  # independent variable mid point
        #         s_lower = s_mid - s_bound[index]
        #         s_upper = s_mid + s_bound[index]
        s_lower = -s_bound[index]
        s_upper = s_bound[index]
        s_lower_2 = s_lower * s_lower
        #         So_2 = s_mid * s_mid
        s_upper_2 = s_upper * s_upper
        mtx_shifted = array(
            [
                [1.0, s_lower, s_lower_2],
                #             [1.0, s_mid, s_mid_2],
                [1.0, 0.0, 0.0],
                [1.0, s_upper, s_upper_2],
            ],
            dtype=self.dtype,
        )
        return mtx_shifted

    @staticmethod
    def __compute_a(mtx_shifted, f_bound, ao_coeff, ai_coeff, aij_coeff, index):
        """Compute the interpolation terms.

        :returns: ao_coeff, ai_coeff, aij_coeff
            constant, linear and quadratic terms
        :rtype: float, 1D array, 2D array (numpy)
        """
        try:
            a_mat = linalg.solve(mtx_shifted, f_bound)
        except linalg.LinAlgError:
            LOGGER.warning(
                "Sobieski polynomial approximation: "
                "exact linear system solve failed, using "
                " approximate least squares method instead."
            )
            a_mat, _, _, _ = lstsq(mtx_shifted, f_bound)

        ao_coeff = a_mat[0]
        ai_coeff[index] = a_mat[1][0]
        aij_coeff[index, index] = a_mat[2][0]
        return ao_coeff, ai_coeff, aij_coeff

    def __update_aij(self, aij_coeff, imax):
        """Update of quadratic interpolation terms.

        :param aij_coeff: array of quadratic terms
        :type aij_coeff: numpy array
        :param imax: size of interpolation matrix
        :type imax: integer
        :returns: aij_coeff (modified quadratic terms)
        :rtype: numpy array
        """
        coeff_mtrix = array(
            [
                [0.2736, 0.3970, 0.8152, 0.9230, 0.1108],
                [0.4252, 0.4415, 0.6357, 0.7435, 0.1138],
                [0.0329, 0.8856, 0.8390, 0.3657, 0.0019],
                [0.0878, 0.7248, 0.1978, 0.0200, 0.0169],
                [0.8955, 0.4568, 0.8075, 0.9239, 0.2525],
            ],
            dtype=self.dtype,
        )

        for i in range(imax):
            for j in range(i + 1, imax):
                aij_coeff[i, j] = aij_coeff[i, i] * coeff_mtrix[i, j]
                aij_coeff[j, i] = aij_coeff[i, j]
        return aij_coeff

    def __compute_fbound(self, flag, s_shifted, a_coeff, b_coeff, index):
        """Compute right-hand side of polynomial function system.

        :param flag: functional relationship between var
        :type flag: numpy array
        :param s_shifted: vector of normalized values of independent variables
                shifted around origin
        :type s_shifted: numpy array
        """
        if flag[index] == 5:
            f_bound = array(
                [
                    [1.0 + (0.25 * a_coeff * a_coeff)],
                    [1.0],
                    [1 + 0.25 * b_coeff * b_coeff],
                ],
                dtype=self.dtype,
            )
        else:
            if flag[index] == 0:
                s_shifted = 0.0
            elif flag[index] == 3:
                a_coeff = -a_coeff
                b_coeff = a_coeff
            elif flag[index] == 2:
                b_coeff = 2 * a_coeff
            elif flag[index] == 4:
                a_coeff = -a_coeff
                b_coeff = 2 * a_coeff
            f_bound = array(
                [[1.0 - (0.5 * a_coeff)], [1.0], [1 + 0.5 * b_coeff]], dtype=self.dtype
            )
        return s_shifted, f_bound

    @staticmethod
    def _normalize_s(s_ref, s_new):
        """Normalization of input variables for use of polynomial approximation.

        :param s_ref: vector of initial values of independent
            variables (5 variables at max)
        :type s_ref: numpy array
        :param s_new: vector of current values of
            independent variables
        :type s_new: numpy array
        :returns: normalized value and normalized+centered value
        :rtype: numpy array
        """
        s_norm = s_new / s_ref
        s_norm[s_norm > 1.25] = 1.25
        s_norm[s_norm < 0.75] = 0.75
        s_shifted = s_norm - 1.0  # Shift S near origin
        return s_norm, s_shifted

    def derive_normalize_s(self, s_ref, s_new):
        """Derivation of normalization of input variables for use of polynomial
        approximation.

        :param s_ref: vector of initial values of independent
            variables (5 variables at max)
        :type s_ref: numpy array
        :param s_new: vector of current values of
            independent variables
        :type s_new: numpy array
        :returns: derivatives of normalized value and normalized+centered value
        :rtype: numpy array
        """
        if self.dtype == complex128:
            s_norm = s_new.real / s_ref.real
        else:
            s_norm = s_new / s_ref

        if s_norm > 1.25:
            derive_s_norm = 0.0
        elif s_norm < 0.75:
            derive_s_norm = 0.0
        else:
            derive_s_norm = 1.0 / s_ref
        #        derive_s_norm = s_new / s_ref
        #        derive_s_norm[derive_s_norm > 1.25] = 0.0
        #        derive_s_norm[derive_s_norm < 0.75] = 0.0
        #        derive_s_norm = derive_s_norm / s_ref

        return derive_s_norm

    def derive_poly_approx(self, s_ref, s_new, flag, s_bound):
        """Compute the polynomial coefficients to characterize the behavior of certain
        synthetic variables and function modifiers. Compared to poly_approx, also
        returns polynomial coeff for linearization.

        :param s_ref: vector of initial values of independent
            variables (5 variables at max)
        :type s_ref: numpy array
        :param s_new: vector of current values of
            independent variables
        :type s_new: numpy array
        :param flag: indicates functional relationship between variables:

            - flag = 1: linear >0
            - flag = 2: nonlinear >0
            - flag = 3: linear < 0
            - flag = 4: nonlinear <0
            - flag = 5: parabolic

        :type flag: int
        :param s_bound: offset value for normalization
        :type s_bound: numpy array
        :returns: poly_value: value of synthetic variable or modifier
        :rtype: numpy array
        """

        imax = s_ref.shape[0]
        # Normalize new S with initial
        _, s_shifted = self._normalize_s(s_ref, s_new)
        #       derive_S_norm = self._derive_normalize_S(s_ref, s_new)

        ao_coeff = 0.0
        ai_coeff = zeros(imax, dtype=self.dtype)
        aij_coeff = zeros((imax, imax), dtype=self.dtype)
        for i in range(imax):
            a_coeff = 0.1
            b_coeff = a_coeff
            mtx_shifted = self.__compute_mtx_shifted(s_bound, i)

            s_shifted, f_bound = self.__compute_fbound(
                flag, s_shifted, a_coeff, b_coeff, i
            )

            ao_coeff, ai_coeff, aij_coeff = self.__compute_a(
                mtx_shifted, f_bound, ao_coeff, ai_coeff, aij_coeff, i
            )

        aij_coeff = self.__update_aij(aij_coeff, imax)

        poly_value = (
            ao_coeff
            + dot(ai_coeff, s_shifted)
            + 0.5 * dot(dot(s_shifted, aij_coeff[:imax, :imax]), s_shifted)
        )
        return poly_value[0], ai_coeff, aij_coeff[:imax, :imax], s_shifted

    def poly_approx(self, s_ref, s_new, flag, s_bound):
        """This function calculates polynomial coefficients to characterize the behavior
        of certain synthetic variables and function modifiers.

        :param s_ref: vector of initial values of independent
            variables (5 variables at max)
        :type s_ref: numpy array
        :param s_new: vector of current values of
            independent variables
        :type s_new: numpy array
        :param flag: indicates functional relationship between variables:

            - flag = 1: linear >0
            - flag = 2: nonlinear >0
            - flag = 3: linear < 0
            - flag = 4: nonlinear <0
            - flag = 5: parabolic

        :type flag: int
        :param s_bound: offset value for normalization
        :type s_bound: numpy array
        :returns: poly_value: value of synthetic variable or modifier
        :rtype: numpy array
        """

        imax = s_ref.shape[0]
        # Normalize new S with initial
        _, s_shifted = self._normalize_s(s_ref, s_new)
        # - ai_coeff: vector of coeff for 2nd term
        #         - aij_coeff: matrix of coeff for 3rd term
        #         - ao_coeff: scalar coeff
        ao_coeff = 0.0
        ai_coeff = zeros(imax, dtype=self.dtype)
        aij_coeff = zeros((imax, imax), dtype=self.dtype)
        for i in range(imax):
            a_coeff = 0.1
            b_coeff = a_coeff
            mtx_shifted = self.__compute_mtx_shifted(s_bound, i)

            s_shifted, f_bound = self.__compute_fbound(
                flag, s_shifted, a_coeff, b_coeff, i
            )

            ao_coeff, ai_coeff, aij_coeff = self.__compute_a(
                mtx_shifted, f_bound, ao_coeff, ai_coeff, aij_coeff, i
            )

        aij_coeff = self.__update_aij(aij_coeff, imax)

        poly_value = (
            ao_coeff
            + dot(ai_coeff, s_shifted)
            + 0.5 * dot(dot(s_shifted, aij_coeff[:imax, :imax]), s_shifted)
        )
        return poly_value[0]

    def compute_half_span(self, x_shared):
        """Compute half-span from surface and aspect ratio.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: half-span
        :rtype: numpy array
        """
        return self.math.sqrt(x_shared[3] * x_shared[5]) * 0.5  # 1/2 span

    def compute_thickness(self, x_shared):
        """Compute a wing thickness.

        :param x_shared: global design variables
        :type x_shared: numpy array
        :returns: thickness
        :rtype: numpy array
        """
        return (
            x_shared[0] * x_shared[5] / (self.math.sqrt(x_shared[5] * x_shared[3]))
        )  # thickness

    @staticmethod
    def compute_aero_center(x_1):
        """Computes the aerodynamic center.

        :param x_1: local design variables (structure)
        :type x_1: numpy array
        :returns: aerodynamic center
        :rtype: numpy array
        """
        return (1.0 + 2.0 * x_1[0]) / (3.0 * (1 + x_1[0]))

    def get_initial_values(self):
        """Get initial values used by polynomial functions.

        :returns: initial values
        :rtype: tuple(ndarray)
        """
        self.i_0 = self.get_default_x0()
        x_initial = self.i_0[1]
        tc_initial = self.i_0[4]
        half_span_initial = self.math.sqrt(self.i_0[7] * self.i_0[9]) * 0.5
        aero_center_initial = (1 + 2 * self.i_0[0]) / (3 * (1 + self.i_0[0]))
        cf_initial = self.i_0[2]
        mach_initial = self.i_0[6]
        h_initial = self.i_0[5]
        throttle_initial = self.i_0[3]
        lift_initial = self.dtype(1.0)
        twist_initial = self.dtype(1.0)
        esf_initial = self.dtype(1.0)
        return (
            x_initial,
            tc_initial,
            half_span_initial,
            aero_center_initial,
            cf_initial,
            mach_initial,
            h_initial,
            throttle_initial,
            lift_initial,
            twist_initial,
            esf_initial,
        )
