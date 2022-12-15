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
"""Propulsion discipline for the Sobieski's SSBJ use case."""
from __future__ import annotations

import logging

from numpy import append
from numpy import array
from numpy import ndarray
from numpy import zeros

from gemseo.problems.sobieski.core.discipline import SobieskiDiscipline
from gemseo.problems.sobieski.core.utils import SobieskiBase

LOGGER = logging.getLogger(__name__)


class SobieskiPropulsion(SobieskiDiscipline):
    """Propulsion discipline for the Sobieski's SSBJ use case."""

    ESF_UPPER_LIMIT = 1.5
    ESF_LOWER_LIMIT = 0.5
    TEMPERATURE_LIMIT = 1.02

    def __init__(self, sobieski_base: SobieskiBase) -> None:
        super().__init__(sobieski_base)
        # Surface fit to engine deck with the least square method
        # Polynomial coefficients for SFC computation
        self.__ao_coeff = zeros(1, dtype=self.dtype)
        self.__ai_coeff = zeros(3, dtype=self.dtype)
        self.__aij_coeff = zeros((3, 3), dtype=self.dtype)
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
        self.__flag_temp = array([2, 4, 2], dtype=self.dtype)
        self.__bound_temp = array([0.25, 0.25, 0.25], dtype=self.dtype)
        self.__s_initial = array(
            [self.mach_initial, self.h_initial, self.throttle_initial], dtype=self.dtype
        )

    def __compute_dim_throttle(
        self,
        adim_throttle: float,
    ) -> float:
        """Compute a dimensioned throttle from an adimensioned one.

        Args:
            adim_throttle: The adimensioned throttle.

        Returns:
            The dimensioned throttle.
        """
        return adim_throttle * self.throttle_coeff

    def __compute_sfc(
        self,
        altitude: float,
        mach: float,
        adim_throttle: float,
    ) -> float:
        """Compute the specific fuel consumption (SFC).

        Args:
            altitude: The altitude.
            mach: The Mach number.
            adim_throttle: The adimensioned throttle.

        Returns:
            The SFC.
        """
        throttle = self.__compute_dim_throttle(adim_throttle)
        return (
            self.sfc_coeff[0]
            + self.sfc_coeff[1] * mach
            + self.sfc_coeff[2] * altitude
            + self.sfc_coeff[3] * throttle
            + self.sfc_coeff[4] * mach**2
            + 2 * altitude * mach * self.sfc_coeff[5]
            + 2 * throttle * mach * self.sfc_coeff[6]
            + self.sfc_coeff[7] * altitude**2
            + 2 * throttle * altitude * self.sfc_coeff[8]
            + self.sfc_coeff[9] * throttle**2
        )

    def __compute_throttle_ua(
        self,
        altitude: float,
        mach: float,
    ) -> float:
        """Compute the throttle upper limit.

        Args:
            altitude: The altitude.
            mach: The Mach number.

        Returns:
            The throttle upper limit.
        """
        return (
            self.thua_coeff[0]
            + self.thua_coeff[1] * mach
            + self.thua_coeff[2] * altitude
            + self.thua_coeff[3] * mach**2
            + 2 * self.thua_coeff[4] * mach * altitude
            + self.thua_coeff[5] * altitude**2
        )

    def __compute_throttle_constraint(
        self,
        altitude: float,
        mach: float,
        adim_throttle: float,
    ) -> float:
        """Compute the throttle constraint.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            adim_throttle: The adimensioned throttle.

        Returns:
            The throttle constraint.
        """
        throttle = self.__compute_dim_throttle(adim_throttle)
        throttle_ua = self.__compute_throttle_ua(altitude, mach)
        return throttle / throttle_ua - 1.0  # throttle setting

    def __compute_dthrconst_dthrottle(
        self,
        altitude: float,
        mach: float,
    ) -> float:
        """Derive the throttle constraint with respect to the throttle.

        Args:
            altitude: The altitude.
            mach: The Mach number.

        Returns:
            The derivative of the throttle constraint with respect to the throttle.
        """
        return self.throttle_coeff / self.__compute_throttle_ua(altitude, mach)

    def __compute_dthrcons_dh(
        self,
        altitude: float,
        mach: float,
        adim_throttle: float,
    ) -> float:
        """Derive the throttle constraint with respect to the altitude.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            adim_throttle: The adimensioned throttle.

        Returns:
            The derivative of the throttle constraint with respect to the altitude.
        """
        throttle = self.__compute_dim_throttle(adim_throttle)
        throttle_ua = self.__compute_throttle_ua(altitude, mach)
        dthrottle_ua_dh = (
            self.thua_coeff[2]
            + 2 * self.thua_coeff[4] * mach
            + 2.0 * self.thua_coeff[5] * altitude
        )
        return -throttle * dthrottle_ua_dh / (throttle_ua * throttle_ua)

    def __compute_dthrconst_dmach(
        self,
        altitude: float,
        mach: float,
        adim_throttle: float,
    ) -> float:
        """Derive the throttle constraint with respect to the Mach number.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            adim_throttle: The adimensioned throttle.

        Returns:
            The derivative of the throttle constraint with respect to the Mach number.
        """
        throttle = self.__compute_dim_throttle(adim_throttle)
        throttle_ua = self.__compute_throttle_ua(altitude, mach)
        dthrottle_ua_dmach = (
            self.thua_coeff[1]
            + 2 * self.thua_coeff[3] * mach
            + 2.0 * self.thua_coeff[4] * altitude
        )
        return -throttle * dthrottle_ua_dmach / (throttle_ua * throttle_ua)

    def __compute_esf(
        self,
        drag: float,
        adim_throttle: float,
    ) -> float:
        """Compute the engine scale factor.

        Args:
            drag: The drag coefficient.
            adim_throttle: The adimensioned throttle.

        Returns:
            The engine scale factor.
        """
        return drag / (3.0 * self.__compute_dim_throttle(adim_throttle))

    def __compute_desf_ddrag(self, adim_throttle: float) -> float:
        """Derive the engine scale factor (ESF) with respect to the drag coefficient.

        Args:
            adim_throttle: The adimensioned throttle.

        Returns:
            The derivative of the ESF with respect to the drag coefficient.
        """
        return 1.0 / (3 * self.__compute_dim_throttle(adim_throttle))

    def __compute_desf_dthrottle(
        self,
        drag: float,
        adim_throttle: float,
    ) -> float:
        """Derive the engine scale factor with respect to the throttle.

        Args:
            drag: The drag coefficient.
            adim_throttle: The adimensioned throttle.

        Returns:
            The derivative of the engin scale factor with respect to the throttle.
        """
        throttle = self.__compute_dim_throttle(adim_throttle)
        return -self.throttle_coeff * drag / (3.0 * throttle**2)

    def __compute_temp(
        self,
        altitude: float,
        mach: float,
        throttle: float,
    ) -> ndarray:
        """Compute the engine temperature.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            throttle: The throttle.

        Return:
            The engine temperature.
        """
        s_new = array([mach, altitude, throttle], dtype=self.dtype)
        return self.base.compute_polynomial_approximation(
            self.__s_initial,
            s_new,
            self.__flag_temp,
            self.__bound_temp,
            self.__ao_coeff,
            self.__ai_coeff,
            self.__aij_coeff,
        )

    def __compute_engine_weight(
        self,
        esf: float,
        c_3: float | None = None,
    ) -> float:
        """Compute the engine weight.

        Args:
            esf: The engine scale factor.
            c_3: The reference engine weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Return:
            The engine weight.
        """
        c_3 = c_3 or self.constants[3]
        return c_3 * (esf**1.05) * 3

    def execute(
        self,
        x_shared: ndarray,
        y_23: ndarray,
        x_3: ndarray,
        true_cstr: bool = False,
        c_3: float | None = None,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        """Compute the fuel consumption, engine weight and engine scale factor.

        Args:
            x_shared: The values of the shared design variables,
                where ``x_shared[0]`` is the thickness/chord ratio,
                ``x_shared[1]`` is the altitude,
                ``x_shared[2]`` is the Mach number,
                ``x_shared[3]`` is the aspect ratio,
                ``x_shared[4]`` is the wing sweep and
                ``x_shared[5]`` is the wing surface area.
            y_23: The drag coefficient.
            x_3: The throttle.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            c_3: The reference engine weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The propulsion outputs:
                - ``y_3``: The outputs of the propulsion analysis:
                    - ``y_3[0]``: the specific fuel consumption,
                    - ``y_3[1]``: the engine weight,
                    - ``y_3[2]``: the engine scale factor,
                - ``g_3``: The propulsion outputs to be constrained:
                    - ``g_3[0]``: the engine scale factor,
                    - ``g_3[1]``: the engine temperature,
                    - ``g_3[2]``: the throttle setting.
        """
        return self._execute(
            x_shared[1],
            x_shared[2],
            x_3[0],
            y_23[0],
            true_cstr=true_cstr,
            ref_weight=c_3,
        )

    def _execute(
        self,
        altitude: float,
        mach: float,
        throttle: float,
        drag: float,
        true_cstr: bool = False,
        ref_weight: float | None = None,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        """Compute the fuel consumption, engine weight and engine scale factor.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            throttle: The throttle.
            drag: The drag coefficient.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            c_3: The reference engine weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The propulsion outputs:
                - ``y_3``: The outputs of the propulsion analysis:
                    - ``y_3[0]``: the specific fuel consumption,
                    - ``y_3[1]``: the engine weight,
                    - ``y_3[2]``: the engine scale factor,
                - ``g_3``: The propulsion outputs to be constrained:
                    - ``g_3[0]``: the engine scale factor,
                    - ``g_3[1]``: the engine temperature,
                    - ``g_3[2]``: the throttle setting.
        """
        c_3 = ref_weight or self.constants[3]
        y_3 = zeros(3, dtype=self.dtype)
        g_3 = zeros(3, dtype=self.dtype)
        y_31 = zeros(1, dtype=self.dtype)
        y_32 = zeros(1, dtype=self.dtype)
        y_34 = zeros(1, dtype=self.dtype)

        y_3[2] = self.__compute_esf(drag, throttle)
        y_3[1] = self.__compute_engine_weight(y_3[2], c_3)
        y_3[0] = self.__compute_sfc(altitude, mach, throttle)

        y_31[0] = y_3[1]
        y_34[0] = y_3[0]
        y_32[0] = y_3[2]

        # THIS SECTION COMPUTES SFC, esf, AND ENGINE WEIGHT
        g_3[0] = y_3[2]  # engine scale factor

        # engine temperature
        g_3[1] = self.__compute_temp(altitude, mach, throttle)

        g_3[2] = self.__compute_throttle_constraint(altitude, mach, throttle)

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

    def __compute_dengineweight_dvar(
        self,
        esf: float,
        desf_dx: ndarray,
        c_3: float | None = None,
    ) -> float:
        """Derive the engine weight with respect to an input variable ``x``.

        Args:
            esf: The engine scale factor (ESF).
            desf_dx: The partial derivative of ESF with respect to an input variable.
            c_3: The reference engine weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The derivative of the engine weight wrt the variable ``x``.
        """
        c_3 = c_3 or self.constants[3]
        return 3 * c_3 * 1.05 * desf_dx * esf**0.05

    def __compute_dsfc_dthrottle(
        self,
        altitude: float,
        mach: float,
        adim_throttle: float,
    ) -> float:
        """Derive the specific fuel consumption constraint with respect to the throttle.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            adim_throttle: The adimensioned throttle.

        Returns:
            The derivative of the SFC constraint with respect to the throttle.
        """
        return (
            self.sfc_coeff[3]
            + 2 * mach * self.sfc_coeff[6]
            + 2 * altitude * self.sfc_coeff[8]
            + 2 * self.sfc_coeff[9] * self.__compute_dim_throttle(adim_throttle)
        ) * self.throttle_coeff

    def __compute_dsfc_dh(
        self,
        altitude: float,
        mach: float,
        adim_throttle: float,
    ) -> float:
        """Derive the specific fuel consumption constraint with respect to the altitude.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            adim_throttle: The adimensioned throttle.

        Returns:
            The derivative of the SFC constraint with respect to the altitude.
        """
        return (
            self.sfc_coeff[2]
            + 2 * mach * self.sfc_coeff[5]
            + +2 * self.sfc_coeff[7] * altitude
            + 2 * self.__compute_dim_throttle(adim_throttle) * self.sfc_coeff[8]
        )

    def __compute_dsfc_dmach(
        self,
        altitude: float,
        mach: float,
        adim_throttle: float,
    ) -> float:
        """Derive the specific fuel consumption constraint wrt the Mach number.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            adim_throttle: The adimensioned throttle.

        Returns:
            The derivative of the SFC constraint with respect to the Mach number.
        """
        return (
            self.sfc_coeff[1]
            + 2 * self.sfc_coeff[4] * mach
            + 2 * altitude * self.sfc_coeff[5]
            + 2 * self.__compute_dim_throttle(adim_throttle) * self.sfc_coeff[6]
        )

    def __dadimthrottle_dthrottle(self, adim_throttle: float) -> float:
        """Derive the adimensioned throttle with respect to the throttle.

        Args:
            adim_throttle: The adimensioned throttle.

        Returns:
            The derivative of the adimensioned throttle with respect to the throttle.
        """
        return self.base.derive_normalization(self.throttle_initial, adim_throttle)

    def __compute_dadimh_dh(self, altitude: float) -> float:
        """Derive the adimensioned throttle with respect to the altitude.

        Args:
            altitude: The altitude.

        Returns:
            The derivative of the adimensioned throttle with respect to the altitude.
        """
        return self.base.derive_normalization(self.h_initial, altitude)

    def __compute_dadimmach_dmach(self, mach: float) -> float:
        """Derive the adimensioned throttle with respect to the Mach number.

        Args:
            mach: The Mach number.

        Returns:
            The derivative of the adimensioned throttle with respect to the Mach number.
        """
        return self.base.derive_normalization(self.mach_initial, mach)

    def __initialize_jacobian(
        self, true_cstr: bool = False
    ) -> dict[str, dict[str, ndarray]]:
        """Initialize the Jacobian structure.

        Args:
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.

        Returns:
            The empty Jacobian structure.
        """
        # Jacobian matrix as a dictionary
        jacobian = {"y_3": {}, "g_3": {}, "y_31": {}, "y_32": {}, "y_34": {}}

        jacobian["y_3"]["x_3"] = zeros((3, 1), dtype=self.dtype)
        jacobian["y_3"]["x_shared"] = zeros((3, 6), dtype=self.dtype)
        jacobian["y_3"]["y_23"] = zeros((3, 1), dtype=self.dtype)
        jacobian["y_3"]["c_3"] = zeros((3, 1), dtype=self.dtype)
        if not true_cstr:
            n_constraints = 4
        else:
            n_constraints = 3
        jacobian["g_3"]["x_3"] = zeros((n_constraints, 1), dtype=self.dtype)
        jacobian["g_3"]["x_shared"] = zeros((n_constraints, 6), dtype=self.dtype)
        jacobian["g_3"]["y_23"] = zeros((n_constraints, 1), dtype=self.dtype)
        jacobian["g_3"]["c_3"] = zeros((n_constraints, 1), dtype=self.dtype)

        return jacobian

    def linearize(
        self,
        x_shared: ndarray,
        y_23: ndarray,
        x_3: ndarray,
        true_cstr: bool = False,
        c_3: float | None = None,
    ) -> dict[str, dict[str, ndarray]]:
        """Derive the fuel consumption, engine weight and engine scale factor.

        Args:
            x_shared: The values of the shared design variables,
                where ``x_shared[0]`` is the thickness/chord ratio,
                ``x_shared[1]`` is the altitude,
                ``x_shared[2]`` is the Mach number,
                ``x_shared[3]`` is the aspect ratio,
                ``x_shared[4]`` is the wing sweep and
                ``x_shared[5]`` is the wing surface area.
            y_23: The drag coefficient.
            x_3: The throttle.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            c_3: The reference engine weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The Jacobian of the discipline.
        """
        return self._linearize(
            x_shared[1],
            x_shared[2],
            x_3[0],
            y_23[0],
            true_cstr=true_cstr,
            ref_weight=c_3,
        )

    def _linearize(
        self,
        altitude: float,
        mach: float,
        throttle: float,
        drag: float,
        true_cstr: bool = False,
        ref_weight: float | None = None,
    ) -> dict[str, dict[str, ndarray]]:
        """Derive the fuel consumption, engine weight and engine scale factor.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            throttle: The throttle.
            drag: The drag coefficient.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            c_3: The reference engine weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The Jacobian of the discipline.
        """

        c_3 = ref_weight or self.constants[3]

        # Jacobian matrix as a dictionary
        jacobian = self.__initialize_jacobian(true_cstr)

        dg_3_dx_3 = zeros((3, 1), dtype=self.dtype)
        dg_3_dxs = zeros((3, 6), dtype=self.dtype)
        dg_3_dy_23 = zeros((3, 1), dtype=self.dtype)

        esf = self.__compute_esf(drag, throttle)
        jacobian["y_3"]["c_3"][1, 0] = 3 * (esf**1.05)

        # dSFC_dthrottle
        jacobian["y_3"]["x_3"][0, 0] = self.__compute_dsfc_dthrottle(
            altitude, mach, throttle
        )

        # dESF_dthrottle
        jacobian["y_3"]["x_3"][2, 0] = self.__compute_desf_dthrottle(drag, throttle)
        # dengineweight_dthrottle
        jacobian["y_3"]["x_3"][1, :] = self.__compute_dengineweight_dvar(
            esf, jacobian["y_3"]["x_3"][2, 0], c_3
        )
        # dSFC_d(t/c) = 0
        # dESF_d(t/c) = 0
        # dengineweight_d(t/c) = 0

        # dSFC_dh
        jacobian["y_3"]["x_shared"][0, 1] = self.__compute_dsfc_dh(
            altitude, mach, throttle
        )
        # dESF_dh= 0.0
        # dengineweight_dh= 0.0

        # dSFC_dM
        jacobian["y_3"]["x_shared"][0, 2] = self.__compute_dsfc_dmach(
            altitude, mach, throttle
        )
        # dESF_dM= 0.0
        # dengineweight_dM= 0.0

        #        jacobian['y_3']['x_shared'][:, 2:] = 0.0

        # dSFC_ddrag
        #        jacobian['y_3']['y_23'][0, 0] = 0.0
        # dESF_ddrag
        jacobian["y_3"]["y_23"][2, 0] = self.__compute_desf_ddrag(throttle)
        # dengineweight_ddrag
        jacobian["y_3"]["y_23"][1, :] = self.__compute_dengineweight_dvar(
            esf, jacobian["y_3"]["y_23"][2, 0]
        )

        # dtemp_ddrag
        #        jacobian['g_3']['y_23'][0, 0] = 0.0
        s_new = array([mach, altitude, throttle], dtype=self.dtype)
        _, ai_coeff, aij_coeff, s_shifted = self.base.derive_polynomial_approximation(
            self.__s_initial,
            s_new,
            self.__flag_temp,
            self.__bound_temp,
            self.__ao_coeff,
            self.__ai_coeff,
            self.__aij_coeff,
        )

        #        dadimtemp_dtemp = self.__compute_dadimtemp_dtemp(x_shared)

        # g_3[0, :] = ESF
        dg_3_dx_3[0, :] = jacobian["y_3"]["x_3"][2, :]
        # dtemp_dthrottle
        dg_3_dx_3[1, 0] = self.__dadimthrottle_dthrottle(throttle) * (
            ai_coeff[2]
            + aij_coeff[2, 0] * s_shifted[0]
            + aij_coeff[2, 1] * s_shifted[1]
            + aij_coeff[2, 2] * s_shifted[2]
        )
        # d(throttle-throttle_ua)_dthrottle
        dg_3_dx_3[2, 0] = self.__compute_dthrconst_dthrottle(
            altitude,
            mach,
        )

        # g_3[0, :] = ESF
        dg_3_dxs[0, :] = jacobian["y_3"]["x_shared"][2, :]

        # dtemp_d(t/c)= 0.0
        # d(throttle-throttle_ua)_d(t/c)= 0.0

        # dtemp_dh
        dg_3_dxs[1, 1] = self.__compute_dadimh_dh(altitude) * (
            ai_coeff[1]
            + aij_coeff[1, 0] * s_shifted[0]
            + aij_coeff[1, 1] * s_shifted[1]
            + aij_coeff[1, 2] * s_shifted[2]
        )
        # d(throttle-throttle_ua)_dh
        dg_3_dxs[2, 1] = self.__compute_dthrcons_dh(altitude, mach, throttle)

        # dtemp_dM
        dg_3_dxs[1, 2] = self.__compute_dadimmach_dmach(mach) * (
            ai_coeff[0]
            + aij_coeff[0, 0] * s_shifted[0]
            + aij_coeff[0, 1] * s_shifted[1]
            + aij_coeff[0, 2] * s_shifted[2]
        )
        # d(throttle-throttle_ua)_dM
        dg_3_dxs[2, 2] = self.__compute_dthrconst_dmach(altitude, mach, throttle)

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
    def __set_coupling_jacobian(
        jacobian,
    ) -> dict[str, dict[str, ndarray]]:
        """Set Jacobian of the coupling variables."""
        jacobian["y_31"]["x_3"] = jacobian["y_3"]["x_3"][1:2, :]
        jacobian["y_31"]["x_shared"] = jacobian["y_3"]["x_shared"][1:2, :]
        jacobian["y_31"]["y_23"] = jacobian["y_3"]["y_23"][1:2, :]
        jacobian["y_31"]["c_3"] = jacobian["y_3"]["c_3"][1:2, :]

        jacobian["y_32"]["x_3"] = jacobian["y_3"]["x_3"][2:3, :]
        jacobian["y_32"]["x_shared"] = jacobian["y_3"]["x_shared"][2:3, :]
        jacobian["y_32"]["y_23"] = jacobian["y_3"]["y_23"][2:3, :]
        jacobian["y_32"]["c_3"] = jacobian["y_3"]["c_3"][2:3, :]

        jacobian["y_34"]["x_3"] = jacobian["y_3"]["x_3"][0:1, :]
        jacobian["y_34"]["x_shared"] = jacobian["y_3"]["x_shared"][0:1, :]
        jacobian["y_34"]["y_23"] = jacobian["y_3"]["y_23"][0:1, :]
        jacobian["y_34"]["c_3"] = jacobian["y_3"]["c_3"][0:1, :]
        return jacobian
