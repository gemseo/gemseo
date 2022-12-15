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
"""Mission discipline for the Sobieski's SSBJ use case."""
from __future__ import annotations

import logging

from numpy import array
from numpy import ndarray
from numpy import zeros

from gemseo.problems.sobieski.core.discipline import SobieskiDiscipline

LOGGER = logging.getLogger(__name__)


class SobieskiMission(SobieskiDiscipline):
    """Mission discipline for the Sobieski's SSBJ use case."""

    @staticmethod
    def __compute_weight_ratio(
        w_t: float,
        w_f: float,
    ) -> float:
        """Compute the weight ratio from the Breguet formula.

        Args:
            w_t: The total weight.
            w_f: The fuel weight.

        Returns:
            The weight ratio ``w_t/(w_t-w_f)``.
        """
        return w_t / (w_t - w_f)

    @staticmethod
    def __compute_dweightratio_dwt(
        w_t: float,
        w_f: float,
    ) -> float:
        """Derive the weight ratio with respect to the total weight.

        Args:
            w_t: The total aircraft weight.
            w_f: The fuel weight.

        Returns:
            The derivative of the weight ratio ``w_t/(w_t-w_f)``
            with respect to the total weight.
        """
        return -w_f / ((w_t - w_f) * (w_t - w_f))

    @staticmethod
    def __compute_dweightratio_dwf(
        w_t: float,
        w_f: float,
    ) -> float:
        """Derive the weight ratio with respect to the fuel weight.

        Args:
            w_t: The total aircraft weight.
            w_f: The fuel weight.

        Returns:
            The derivative of the weight ratio ``w_t/(w_t-w_f)``
            with respect to the fuel weight.
        """
        return w_t / ((w_t - w_f) * (w_t - w_f))

    def __compute_dlnweightratio_dwt(
        self,
        w_t: float,
        w_f: float,
    ) -> float:
        """Derive the logarithm of the weight ratio with respect to the total weight.

        Args:
            w_t: The total aircraft weight.
            w_f: The fuel weight.

        Returns:
            The derivative of the logarithm of the weight ratio ``w_t/(w_t-w_f)``
            with respect to the total weight.
        """
        return self.__compute_dweightratio_dwt(w_t, w_f) / self.__compute_weight_ratio(
            w_t, w_f
        )

    def __compute_dlnweightratio_dwf(
        self,
        w_t: float,
        w_f: float,
    ) -> float:
        """Derive the logarithm of the weight ratio with respect to the fuel weight.

        Args:
            w_t: The total aircraft weight.
            w_f: The fuel weight.

        Returns:
            The derivative of the logarithm of the weight ratio ``w_t/(w_t-w_f)``
            with respect to the fuel weight.
        """
        return self.__compute_dweightratio_dwf(w_t, w_f) / self.__compute_weight_ratio(
            w_t, w_f
        )

    def __compute_range(
        self,
        altitude: float,
        mach: float,
        w_t: float,
        w_f: float,
        cl_cd: float,
        sfc: float,
    ) -> float:
        """Compute the range from the Breguet formula.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            w_t: The total aircraft weight.
            w_f: The fuel weight.
            cl_cd: The lift-over-drag ratio.
            sfc: The specific fuel consumption.

        Returns:
            The range.
        """
        sqrt_theta = self.__compute_sqrt_theta(altitude)
        return ((mach * cl_cd) * 661.0 * sqrt_theta / sfc) * self.math.log(
            w_t / (w_t - w_f)
        )

    def __compute_drange_dtotalweight(
        self,
        mach: float,
        w_t: float,
        w_f: float,
        cl_cd: float,
        sfc: float,
        sqrt_theta: float,
    ) -> float:
        """Derive the range with respect to the total weight.

        Args:
            mach: The Mach number.
            w_t: The total aircraft weight.
            w_f: The fuel weight.
            cl_cd: The lift-over-drag ratio.
            sfc: The specific fuel consumption.
            sqrt_theta: The square root of the air temperature.

        Returns:
            The derivative of the range with respect to the total weight.
        """
        return (
            mach
            * cl_cd
            / sfc
            * 661.0
            * sqrt_theta
            * self.__compute_dlnweightratio_dwt(w_t, w_f)
        )

    def __compute_drange_dfuelweight(
        self,
        mach: float,
        w_t: float,
        w_f: float,
        cl_cd: float,
        sfc: float,
        sqrt_theta: float,
    ) -> float:
        """Derive the range with respect to the fuel weight.

        Args:
            mach: The Mach number.
            w_t: The total aircraft weight.
            w_f: The fuel weight.
            cl_cd: The lift-over-drag ratio.
            sfc: The specific fuel consumption.
            sqrt_theta: The square root of the air temperature.

        Returns:
            The derivative of the range with respect to the fuel weight.
        """
        return (
            mach
            * cl_cd
            / sfc
            * 661.0
            * sqrt_theta
            * self.__compute_dlnweightratio_dwf(w_t, w_f)
        )

    @staticmethod
    def __compute_dtheta_dh(altitude: float) -> float:
        """Derive the square root of the air temperature wrt the altitude.

        Args:
            altitude: The altitude of the aircraft.

        Returns:
            The derivative of the square root of the air temperature wrt the altitude.
        """
        if altitude < 36089.0:
            return -0.000006875
        else:
            return 0.0

    def __compute_sqrt_theta(self, altitude: float) -> float:
        """Compute the square root of the air temperature.

        Args:
            altitude: The altitude of the aircraft.

        Returns:
            The square root of the air temperature.
        """
        if altitude < 36089.0:
            return self.math.sqrt(1 - 0.000006875 * altitude)
        else:
            return self.math.sqrt(0.7519)

    def execute(
        self,
        x_shared: ndarray,
        y_14: ndarray,
        y_24: ndarray,
        y_34: ndarray,
    ) -> ndarray:
        """Compute the range.

        Args:
            x_shared: The shared design variables.
            y_14: The total aircraft weight ``y_14[0]`` and the fuel weight ``y_14[1]``.
            y_24: The lift-over-drag ratio.
            y_34: The specific fuel consumption.

        Returns:
            The range.
        """
        return self._execute(
            x_shared[1], x_shared[2], y_14[0], y_14[1], y_24[0], y_34[0]
        )

    def _execute(
        self,
        altitude: float,
        mach: float,
        w_t: float,
        w_f: float,
        cl_cd: float,
        sfc: float,
    ) -> ndarray:
        """Compute the range.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            w_t: The total aircraft weight.
            w_f: The fuel weight.
            cl_cd: The lift-over-drag ratio.
            sfc: The specific fuel consumption.

        Returns:
            The range.
        """
        return array(
            [self.__compute_range(altitude, mach, w_t, w_f, cl_cd, sfc)],
            dtype=self.dtype,
        )

    def __initialize_jacobian(self) -> dict[str, dict[str, ndarray]]:
        """Initialize the Jacobian.

        Returns:
            The empty Jacobian.
        """
        jacobian = {"y_4": {}}
        jacobian["y_4"]["x_shared"] = zeros((1, 6), dtype=self.dtype)
        jacobian["y_4"]["y_14"] = zeros((1, 2), dtype=self.dtype)
        jacobian["y_4"]["y_24"] = zeros((1, 1), dtype=self.dtype)
        jacobian["y_4"]["y_34"] = zeros((1, 1), dtype=self.dtype)
        return jacobian

    def linearize(
        self,
        x_shared: ndarray,
        y_14: ndarray,
        y_24: ndarray,
        y_34: ndarray,
    ) -> dict[str, dict[str, ndarray]]:
        """Derive the discipline with respect to its inputs.

        Args:
            x_shared: The shared design variables.
            y_14: The total aircraft weight ``y_14[0]`` and the fuel weight ``y_14[1]``.
            y_24: The lift-over-drag ratio.
            y_34: The specific fuel consumption.

        Returns:
            The Jacobian of the discipline.
        """
        return self._linearize(
            x_shared[1], x_shared[2], y_14[0], y_14[1], y_24[0], y_34[0]
        )

    def _linearize(
        self,
        altitude: float,
        mach: float,
        w_t: float,
        w_f: float,
        cl_cd: float,
        sfc: float,
    ) -> dict[str, dict[str, ndarray]]:
        """Derive the discipline with respect to its inputs.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            w_t: The total aircraft weight.
            w_f: The fuel weight.
            cl_cd: The lift-over-drag ratio.
            sfc: The specific fuel consumption.

        Returns:
            The Jacobian of the discipline.
        """

        jacobian = self.__initialize_jacobian()
        sqrt_theta = self.__compute_sqrt_theta(altitude)
        dtheta_dh = self.__compute_dtheta_dh(altitude)
        ac_range = self.__compute_range(altitude, mach, w_t, w_f, cl_cd, sfc)

        # dR_d(h)
        jacobian["y_4"]["x_shared"][0, 1] = (
            0.5 * ac_range * dtheta_dh / (sqrt_theta * sqrt_theta)
        )

        # dR_dM
        jacobian["y_4"]["x_shared"][0, 2] = ac_range / mach

        # dR_dWt
        jacobian["y_4"]["y_14"][0, 0] = self.__compute_drange_dtotalweight(
            mach, w_t, w_f, cl_cd, sfc, sqrt_theta
        )
        # dR_dWf
        jacobian["y_4"]["y_14"][0, 1] = self.__compute_drange_dfuelweight(
            mach, w_t, w_f, cl_cd, sfc, sqrt_theta
        )

        # dR_d(L/D)
        jacobian["y_4"]["y_24"][0, 0] = ac_range / cl_cd

        # dR_d(SFC)
        jacobian["y_4"]["y_34"][0, 0] = -ac_range / sfc
        return jacobian
