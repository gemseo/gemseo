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
"""Sobieski's SSBJ base class."""
from __future__ import annotations

import cmath
import logging
import math
from typing import Sequence

from numpy import array
from numpy import atleast_2d
from numpy import clip
from numpy import complex128
from numpy import concatenate
from numpy import dot
from numpy import float64
from numpy import ndarray

LOGGER = logging.getLogger(__name__)


DEG_TO_RAD = math.pi / 180.0
# DO NOT CHANGE THE DEFAULT DESIGN VALUE: IT IS USED FOR POLYNOMIAL APPROXIMATION
_DEFAULT_DESIGN = array([0.25, 1.0, 1.0, 0.5, 0.05, 45000.0, 1.6, 5.5, 55.0, 1000.0])
_MINIMUM_FUEL_WEIGHT = 2000.0
_MISC_WEIGHT = 25000.0
_MAXIMUM_LOAD_FACTOR = 6.0
_REFERENCE_ENGINE_WEIGHT = 4360.0
_MINIMUM_DRAG_COEFFICIENT = 0.01375
_CONSTANTS = array(
    [
        _MINIMUM_FUEL_WEIGHT,
        _MISC_WEIGHT,
        _MAXIMUM_LOAD_FACTOR,
        _REFERENCE_ENGINE_WEIGHT,
        _MINIMUM_DRAG_COEFFICIENT,
    ]
)
_DESIGN_BOUNDS = array(
    [
        (0.1, 0.4),
        (0.75, 1.25),
        (0.75, 1.25),
        (0.1, 1),
        (0.01, 0.09),
        (30000.0, 60000.0),
        (1.4, 1.8),
        (2.5, 8.5),
        (40.0, 70.0),
        (500.0, 1500.0),
    ]
)
_NAMES_TO_BOUNDS = {
    "x_1": _DESIGN_BOUNDS[0:2],
    "x_2": _DESIGN_BOUNDS[2],
    "x_3": _DESIGN_BOUNDS[3],
    "x_shared": _DESIGN_BOUNDS[4:10],
    "y_14": array([(4.97e04 * 0.5, 5.14e04 * 1.5), (-1.54e04 * 0.5, 3.0e04 * 1.5)]),
    "y_12": array([(4.97e04 * 0.5, 5.15e04 * 1.5), (0.9 * 0.5, 1.00)]),
    "y_21": array([(4.97e04 * 0.5, 5.15e04 * 1.5)]),
    "y_23": array([(6.73e03 * 0.5, 1.76e04 * 1.5)]),
    "y_24": array([(0.88 * 0.5, 7.42 * 1.5)]),
    "y_31": array([(5.92e03 * 0.5, 6.79e03 * 1.5)]),
    "y_32": array([(0.47 * 0.5, 0.53 * 1.5)]),
    "y_34": array([(0.88 * 0.5, 1.32 * 1.5)]),
}


class SobieskiBase:
    """Utilities for Sobieski's SSBJ use case."""

    DTYPE_COMPLEX = "complex128"
    DTYPE_DOUBLE = "float64"
    DTYPE_DEFAULT = DTYPE_COMPLEX

    def __init__(
        self,
        dtype: str,
    ) -> None:
        """
        Args:
            dtype: The NumPy data type.
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
            raise ValueError(f"Unknown dtype: {dtype}.")

        self.__constants = _CONSTANTS.astype(self.dtype)
        self.__coeff_mtrix = array(
            [
                [0.2736, 0.3970, 0.8152, 0.9230, 0.1108],
                [0.4252, 0.4415, 0.6357, 0.7435, 0.1138],
                [0.0329, 0.8856, 0.8390, 0.3657, 0.0019],
                [0.0878, 0.7248, 0.1978, 0.0200, 0.0169],
                [0.8955, 0.4568, 0.8075, 0.9239, 0.2525],
            ],
            dtype=self.dtype,
        )
        self.__mtx_shifted = array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=self.dtype,
        )
        self.__f_bound = array(
            [[1.0], [1.0], [1.0]],
            dtype=self.dtype,
        )

        self.__initial_design = _DEFAULT_DESIGN.astype(self.dtype)

    @property
    def initial_design(self) -> ndarray:
        """The initial design."""
        return self.__initial_design

    @property
    def constants(self) -> ndarray:
        """The default constants."""
        return self.__constants

    @property
    def design_bounds(self) -> tuple[ndarray, ndarray]:
        """The lower and upper bounds of the design variables."""
        return _DESIGN_BOUNDS[:, 0], _DESIGN_BOUNDS[:, 1]

    @classmethod
    def get_bounds_by_name(
        cls,
        variables_names: str | Sequence[str],
    ) -> tuple[ndarray, ndarray]:
        """Return the bounds of the design and coupling variables.

        Args:
            variables_names: The names of the variables.

        Returns:
            The lower and upper bounds of these variables.
        """
        if isinstance(variables_names, str):
            variables_names = [variables_names]

        bounds = atleast_2d(
            concatenate(
                [_NAMES_TO_BOUNDS[variable_name] for variable_name in variables_names],
                axis=0,
            )
        )
        return bounds[:, 0], bounds[:, 1]

    def __compute_mtx_shifted(
        self,
        s_bound: ndarray,
        index: int,
    ) -> None:
        """Compute a matrix of shifted values of the design variables.

        Args:
            s_bound: The bounds used to control the slope of the polynomial function.
            index: The index of the design variable in the polynomial function.
        """
        self.__mtx_shifted[0, 1] = -s_bound[index]
        self.__mtx_shifted[2, 1] = s_bound[index]
        self.__mtx_shifted[0, 2] = self.__mtx_shifted[2, 2] = s_bound[index] ** 2

    @staticmethod
    def __compute_a(
        mtx_shifted: ndarray,
        f_bound: ndarray,
        ao_coeff: ndarray,
        ai_coeff: ndarray,
        aij_coeff: ndarray,
        index: int,
    ) -> None:
        """Compute the interpolation terms.

        Args:
            mtx_shifted: The shift matrix.
            f_bound: The f bound.
            ao_coeff: The a0 term.
            ai_coeff: The ai terms
            aij_coeff: The aij term
            index: The index.
        """
        ao_coeff[:] = f_bound[1]
        ai_coeff[index] = -(f_bound[2] - f_bound[0]) / (2 * mtx_shifted[0, 1])
        aij_coeff[index, index] = (
            f_bound[0] - f_bound[1] + (f_bound[2] - f_bound[0]) / 2
        ) / mtx_shifted[0, 2]

    def __update_aij(
        self,
        aij_coeff: ndarray,
        imax: int,
    ) -> ndarray:
        """Update the quadratic interpolation terms.

        Args:
            aij_coeff: The quadratic terms.
            imax: The size of the interpolation matrix.

        Returns:
            The modified quadratic terms.
        """
        for i in range(imax):
            for j in range(i + 1, imax):
                aij_coeff[i, j] = aij_coeff[i, i] * self.__coeff_mtrix[i, j]
                aij_coeff[j, i] = aij_coeff[i, j]

        return aij_coeff

    def __compute_fbound(
        self,
        flag: ndarray,
        s_shifted: ndarray,
        a_coeff: ndarray,
        b_coeff: ndarray,
        index: int,
    ) -> ndarray:
        """Compute right-hand side of polynomial function system.

        Args:
            flag: The functional relationship between variables.
            s_shifted: The normalized values of the independent variables
                shifted around the origin.
            a_coeff: The a coefficient.
            b_coeff: The b coefficient.
            index: The index.

        Returns:
            The right-hand side of polynomial function system.
        """
        if flag[index] == 5:
            self.__f_bound[0, 0] = 1.0 + 0.25 * a_coeff * a_coeff
            self.__f_bound[2, 0] = 1.0 + 0.25 * b_coeff * b_coeff
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

            self.__f_bound[0, 0] = 1.0 - 0.5 * a_coeff
            self.__f_bound[2, 0] = 1.0 + 0.5 * b_coeff

        return s_shifted

    @staticmethod
    def _compute_normalization(
        s_ref: ndarray,
        s_new: ndarray,
    ) -> ndarray:
        """Normalization of input variables for polynomial approximation.

        Args:
            s_ref: The initial values of the independent variables (5 variables at max).
            s_new: The current values of the independent variables.

        Returns:
            The normalized and centered value.
        """
        return clip(s_new / s_ref, 0.75, 1.25) - 1.0

    @staticmethod
    def derive_normalization(
        s_ref: float | ndarray,
        s_new: float | ndarray,
    ) -> float | ndarray:
        """Derivation of normalization of input variables.

        For use of polynomial approximation.

        Args:
            s_ref: The initial values of the independent variables (5 variables at max).
            s_new: The current values of the independent variables.

        Returns:
            Derivatives of normalized value.
        """
        s_norm = s_new.real / s_ref.real
        if s_norm > 1.25:
            return 0.0
        elif s_norm < 0.75:
            return 0.0
        else:
            return 1.0 / s_ref

    def derive_polynomial_approximation(
        self,
        s_ref: ndarray,
        s_new: ndarray,
        flag: int,
        s_bound: ndarray,
        a0_coeff: ndarray,
        ai_coeff: ndarray,
        aij_coeff: ndarray,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """Compute the polynomial coefficients for both evaluation and linearization.

        These coefficients characterize
        the behavior of certain synthetic variables and function modifiers.

        Args:
            s_ref: The initial values of the independent variables (5 variables at max).
            s_new: The current values of the independent variables.
            flag: The functional relationship between the variables:
                - flag = 1: linear >0,
                - flag = 2: nonlinear >0,
                - flag = 3: linear < 0,
                - flag = 4: nonlinear <0,
                - flag = 5: parabolic.
            s_bound: The offset value for normalization.

        Returns:
            The value of the synthetic variables or function modifiers
            for both evaluation and linearization.
        """
        imax = s_ref.shape[0]
        # Normalize new S with initial
        s_shifted = self._compute_normalization(s_ref, s_new)

        a0_coeff[...] = 0
        ai_coeff[...] = 0
        aij_coeff[...] = 0
        for i in range(imax):
            a_coeff = 0.1
            b_coeff = a_coeff
            self.__compute_mtx_shifted(s_bound, i)
            s_shifted = self.__compute_fbound(flag, s_shifted, a_coeff, b_coeff, i)
            self.__compute_a(
                self.__mtx_shifted, self.__f_bound, a0_coeff, ai_coeff, aij_coeff, i
            )

        aij_coeff = self.__update_aij(aij_coeff, imax)

        poly_value = (
            a0_coeff
            + dot(ai_coeff, s_shifted)
            + 0.5 * dot(dot(s_shifted, aij_coeff[:imax, :imax]), s_shifted)
        )
        return poly_value[0], ai_coeff, aij_coeff[:imax, :imax], s_shifted

    def compute_polynomial_approximation(
        self,
        s_ref: ndarray,
        s_new: ndarray,
        flag: int,
        s_bound: ndarray,
        a0_coeff: ndarray,
        ai_coeff: ndarray,
        aij_coeff: ndarray,
    ) -> ndarray:
        """Compute the polynomial coefficients.

        These coefficients characterize
        the behavior of certain synthetic variables and function modifiers.

        Args:
            s_ref: The initial values of the independent variables (5 variables at max).
            s_new: The current values of the independent variables.
            flag: The functional relationship between the variables:
                - flag = 1: linear >0,
                - flag = 2: nonlinear >0,
                - flag = 3: linear < 0,
                - flag = 4: nonlinear <0,
                - flag = 5: parabolic.
            s_bound: The offset value for normalization.

        Returns:
            The value of the synthetic variables or function modifiers.
        """
        imax = s_ref.shape[0]
        # Normalize new S with initial
        s_shifted = self._compute_normalization(s_ref, s_new)
        # - ai_coeff: vector of coeff for 2nd term
        #         - aij_coeff: matrix of coeff for 3rd term
        #         - ao_coeff: scalar coeff
        a0_coeff[...] = 0
        ai_coeff[...] = 0
        aij_coeff[...] = 0
        for i in range(imax):
            a_coeff = 0.1
            b_coeff = a_coeff
            self.__compute_mtx_shifted(s_bound, i)
            s_shifted = self.__compute_fbound(flag, s_shifted, a_coeff, b_coeff, i)
            self.__compute_a(
                self.__mtx_shifted, self.__f_bound, a0_coeff, ai_coeff, aij_coeff, i
            )

        aij_coeff = self.__update_aij(aij_coeff, imax)

        poly_value = (
            a0_coeff
            + dot(ai_coeff, s_shifted)
            + 0.5 * dot(dot(s_shifted, aij_coeff[:imax, :imax]), s_shifted)
        )
        return poly_value[0]

    def compute_half_span(
        self,
        aspect_ratio: float,
        wing_surface_area: float,
    ) -> float:
        """Compute the half-span from the wing surface and aspect ratio.

        Args:
            aspect_ratio: The aspect ratio.
            wing_surface_area: The wing_surface_area.

        Returns:
            The half-span.
        """
        return self.math.sqrt(aspect_ratio * wing_surface_area) * 0.5

    def compute_thickness(
        self,
        aspect_ratio: float,
        thickness_to_chord_ratio: float,
        wing_surface_area: float,
    ) -> float:
        """Compute a wing thickness.

        Args:
            aspect_ratio: The aspect ratio.
            thickness_to_chord_ratio: The thickness to chord ratio.
            wing_surface_area: The wing_surface_area.

        Returns:
            The wing thickness.
        """
        return (
            thickness_to_chord_ratio
            * wing_surface_area
            / (self.math.sqrt(aspect_ratio * wing_surface_area))
        )

    @staticmethod
    def compute_aero_center(
        wing_taper_ratio: float,
    ) -> float:
        """Computes the aerodynamic center.

        Args:
            wing_taper_ratio: The wing taper ratio.

        Returns:
            The aerodynamic center.
        """
        return (1.0 + 2.0 * wing_taper_ratio) / (3.0 * (1 + wing_taper_ratio))

    def get_initial_values(self) -> tuple[float]:
        """Return the initial values used by the polynomial functions.

        Returns:
            The initial values used by the polynomial functions.
        """
        x_initial = self.initial_design[1]
        tc_initial = self.initial_design[4]
        half_span_initial = (
            self.math.sqrt(self.initial_design[7] * self.initial_design[9]) * 0.5
        )
        aero_center_initial = (1 + 2 * self.initial_design[0]) / (
            3 * (1 + self.initial_design[0])
        )
        cf_initial = self.initial_design[2]
        mach_initial = self.initial_design[6]
        h_initial = self.initial_design[5]
        throttle_initial = self.initial_design[3]
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
