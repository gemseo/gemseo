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
"""Aerodynamics discipline for the Sobieski's SSBJ use case."""
from __future__ import annotations

import logging
from math import pi

from numpy import array
from numpy import cos
from numpy import ndarray
from numpy import sin
from numpy import sqrt
from numpy import zeros

from gemseo.problems.sobieski.core.discipline import SobieskiDiscipline
from gemseo.problems.sobieski.core.utils import DEG_TO_RAD
from gemseo.problems.sobieski.core.utils import SobieskiBase

LOGGER = logging.getLogger(__name__)


class SobieskiAerodynamics(SobieskiDiscipline):
    """Aerodynamics discipline for the Sobieski's SSBJ use case."""

    PRESSURE_GRADIENT_LIMIT = 1.04

    def __init__(self, sobieski_base: SobieskiBase) -> None:
        super().__init__(sobieski_base)
        self.__flag1 = array([1, 1], dtype=self.dtype)
        self.__bound1 = array([0.25, 0.25], dtype=self.dtype)
        self.__flag2 = array([5], dtype=self.dtype)
        self.__bound2 = array([0.25], dtype=self.dtype)
        self.__flag3 = array([1], dtype=self.dtype)
        self.__bound3 = array([0.25], dtype=self.dtype)
        self.__ao_coeff_1 = zeros(1, dtype=self.dtype)
        self.__ai_coeff_1 = zeros(2, dtype=self.dtype)
        self.__aij_coeff_1 = zeros((2, 2), dtype=self.dtype)
        self.__ao_coeff_2 = zeros(1, dtype=self.dtype)
        self.__ai_coeff_2 = zeros(1, dtype=self.dtype)
        self.__aij_coeff_2 = zeros((1, 1), dtype=self.dtype)
        self.__ao_coeff_3 = zeros(1, dtype=self.dtype)
        self.__ai_coeff_3 = zeros(1, dtype=self.dtype)
        self.__aij_coeff_3 = zeros((1, 1), dtype=self.dtype)
        self.__esf_cf_initial = array(
            [self.esf_initial, self.cf_initial], dtype=self.dtype
        )
        self.__twist_initial = array([self.twist_initial], dtype=self.dtype)
        self.__tc_initial = array([self.tc_initial], dtype=self.dtype)
        self.__rho = None
        self.__velocity = None
        self.__rhov2 = None
        self.__k_aero = None
        self.__lift_coeff = None

    def __compute_k_aero(
        self,
        mach: float,
        sweep: float,
    ) -> float:
        """Compute the induced drag coefficient (related to lift).

        Args:
            mach: The Mach number.
            sweep: The wing sweep.

        Returns:
            The induced drag coefficient.
        """
        self.__k_aero = (
            (mach**2 - 1)
            * self.math.cos(sweep * DEG_TO_RAD)
            / (4.0 * self.math.sqrt(sweep**2 - 1) - 2)
        )
        return self.__k_aero

    @staticmethod
    def __compute_dk_aero_dsweep(
        mach: float,
        sweep: float,
    ) -> float:
        """Derive the induced drag coefficient with respect to the sweep.

        Args:
            mach: The Mach number.
            sweep: The wing sweep.

        Returns:
            The derivative of the induced drag coefficient with respect to the sweep.
        """
        u_velo = (mach * mach - 1.0) * cos(sweep * DEG_TO_RAD)
        up_velo = -DEG_TO_RAD * (mach * mach - 1.0) * sin(sweep * DEG_TO_RAD)
        v_velo = 4.0 * sqrt(sweep * sweep - 1.0) - 2.0
        vp_velo = 4.0 * sweep * (sweep * sweep - 1.0) ** -0.5
        return (up_velo * v_velo - u_velo * vp_velo) / v_velo**2

    @staticmethod
    def __compute_dk_aero_dmach(
        mach: float,
        sweep: float,
    ) -> float:
        """Derive the induced drag coefficient with respect to the Mach number.

        Args:
            mach: The Mach number.
            sweep: The wing sweep.

        Returns:
            The derivative of the induced drag coefficient
            with respect to the Mach number.
        """
        return (
            (2.0 * mach)
            * cos(sweep * pi / 180.0)
            / (4.0 * sqrt(sweep**2 - 1.0) - 2.0)
        )

    def __compute_dadimcf_dcf(
        self,
        c_f: float,
    ) -> float:
        """Derive the adimensional friction coefficient with respect to the friction
        coefficient.

        Args:
            c_f: The skin friction coefficient.

        Returns:
            The derivative of the adimensional friction coefficient
            with respect to the friction coefficient.
        """
        return self.base.derive_normalization(self.cf_initial, c_f)

    def __compute_dadimtwist_dtwist(
        self,
        twist: float,
    ) -> float:
        """Derive the adimensional twist with respect to the twist.

        Args:
            twist: The wing twist.

        Returns:
            The derivative of the adimensional twist
            with respect to the twist.
        """
        return self.base.derive_normalization(self.twist_initial, twist)

    def __compute_dadimesf_desf(
        self,
        esf: float,
    ) -> float:
        """Derivate the adimensional ESF with respect to the ESF.

        Args:
            esf: The engine scale factor.

        Returns:
            The derivative of the adimensional ESF
            with respect to the ESF.
        """
        return self.base.derive_normalization(self.esf_initial, esf)

    def __compute_dadimtaper_dtaper(
        self,
        tc_ratio: float,
    ) -> float:
        """Derive an adimensional taper-ratio of polynomial with respect to taper-ratio.

        Args:
            tc_ratio: The thickness-to-chord ratio.

        Returns:
            The derivative of the adimensioned taper-ratio of polynomial
            with respect to the taper-ratio.
        """
        return self.base.derive_normalization(self.tc_initial, tc_ratio)

    def __compute_cd_min(
        self,
        tc_ratio: float,
        sweep: float,
        fo1: float,
        c_4: float | None = None,
    ) -> float:
        """Compute the 2D minimum drag coefficient.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            sweep: The wing sweep.
            fo1: The coefficient for the engine size.
            c_4: The minimum drag coefficient.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The 2D minimum drag coefficient.
        """
        c_4 = c_4 or self.constants[4]
        return (
            c_4 * fo1
            + 3.05
            * tc_ratio ** (5.0 / 3.0)
            * (self.math.cos(sweep * DEG_TO_RAD)) ** 1.5
        )

    @staticmethod
    def __compute_dcdmin_dsweep(
        tc_ratio: float,
        sweep: float,
    ) -> float:
        """Derive the 2D minimum drag coefficient with respect to the sweep.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            sweep: The wing sweep.

        Returns:
            The derivative of the 2D minimum drag coefficient
            with respect to the sweep.
        """
        ang_rad = sweep * DEG_TO_RAD
        return (
            -3.05
            * 1.5
            * tc_ratio ** (5.0 / 3.0)
            * cos(ang_rad) ** 0.5
            * DEG_TO_RAD
            * sin(ang_rad)
        )

    def __compute_cd(
        self,
        lift_coeff: float,
        fo2: float,
        cdmin: float,
    ) -> float:
        """Compute of total drag coefficient.

        Args:
            lift_coeff: The lift coefficient.
            fo2: The coefficient for the twist influence on drag.
            cdmin: The 2D minimum drag coefficient.

        Returns:
            The drag coefficient.
        """
        return fo2 * (cdmin + self.__k_aero * lift_coeff * lift_coeff)

    def __compute_dcd_dsweep(
        self,
        tc_ratio: float,
        mach: float,
        sweep: float,
        lift_coeff: float,
        fo2: float,
    ) -> float:
        """Derive the drag coefficient with respect to the sweep.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            mach: The Mach number.
            sweep: The wing sweep.
            lift_coeff: The lift coefficient.
            fo2: The coefficient for the twist influence on the drag.

        Returns:
            The derivative of the drag coefficient with respect to the sweep.
        """
        dcdmin_dsweep = self.__compute_dcdmin_dsweep(tc_ratio, sweep)
        dk_dsweep = self.__compute_dk_aero_dsweep(mach, sweep)
        return fo2 * (dcdmin_dsweep + lift_coeff * lift_coeff * dk_dsweep)

    def __compute_dcd_dsref(
        self,
        wing_area: float,
        fo2: float,
    ) -> float:
        """Derive the drag coefficient with respect to the reference surface.

        Args:
            wing_area: The wing surface area.
            fo2: The coefficient for the twist influence on the drag.

        Returns:
            The derivative of the drag coefficient
            with respect to the reference surface.
        """
        dcl_dsref = self.__compute_dcl_dsref(wing_area)
        return 2 * self.__k_aero * self.__lift_coeff * dcl_dsref * fo2

    def __compute_dcd_dmach(
        self,
        altitude: float,
        mach: float,
        sweep: float,
        wing_area: float,
        ac_mass: float,
        fo2: float,
    ) -> float:
        """Derive the drag coefficient with respect to Mach number.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            sweep: The wing sweep.
            wing_area: The wing surface area.
            ac_mass: The total aicraft mass.
            fo2: The coefficient for the twist influence on the drag.

        Returns:
            The derivative of the drag coefficient with respect to the Mach number.
        """
        dk_dmach = self.__compute_dk_aero_dmach(mach, sweep)
        dcl_dmach = self.__compute_dcl_dmach(altitude, wing_area, ac_mass)
        return (
            (2.0 * self.__k_aero * dcl_dmach + self.__lift_coeff * dk_dmach)
            * self.__lift_coeff
            * fo2
        )

    def __compute_cl(
        self,
        wing_area: float,
        ac_mass: float,
    ) -> float:
        """Computation of the lift coefficient.

        Args:
            wing_area: The wing surface area.
            ac_mass: The total aircraft mass.

        Returns:
            The lift coefficient.
        """
        self.__lift_coeff = self.__compute_adim_coeff(ac_mass, wing_area)
        return self.__lift_coeff

    def __compute_dcl_dh(
        self,
        altitude: float,
        mach: float,
        wing_area: float,
        ac_mass: float,
    ) -> float:
        """Derive the lift coefficient with respect to the altitude.

        Args:
            altitude: The altitude.
            mach: The Mach number.
            wing_area: The wing surface area.
            ac_mass: The total aicraft mass.

        Returns:
            The derivative of the lift coefficient with respect to the altitude.
        """
        drhov2_dh = self.__compute_drhov2_dh(altitude, mach)
        return -2 * ac_mass / wing_area * drhov2_dh / (self.__rhov2**2)

    def __compute_dcl_dmach(
        self,
        altitude: float,
        wing_area: float,
        ac_mass: float,
    ) -> float:
        """Derive the lift coefficient with respect to the Mach number.

        Args:
            altitude: The altitude.
            wing_area: The wing surface area.
            ac_mass: The total aircraft mass.

        Returns:
            The derivative of the lift coefficient with respect to the Mach number.
        """
        dv_dmach = self.__compute_dv_dmach(altitude)
        return (
            -4.0
            * ac_mass
            * dv_dmach
            / (
                self.__rho
                * wing_area
                * self.__velocity
                * self.__velocity
                * self.__velocity
            )
        )

    def __compute_dcl_dsref(
        self,
        wing_area: float,
    ) -> float:
        """Derive the lift coefficient with respect to the reference surface.

        Args:
            wing_area: The wing surface area.

        Returns:
            The derivative of the lift coefficient
            with respect to the reference surface.
        """
        return -self.__lift_coeff / wing_area

    def __compute_rhov2(self) -> float:
        """Compute :math:`\rho v^2` (2*dynamic pressure).

        Returns:
            :math:`\rho v^2`.
        """
        self.__rhov2 = self.__rho * self.__velocity * self.__velocity
        return self.__rhov2

    def __compute_drhov2_dh(
        self,
        altitude: float,
        mach: float,
    ) -> float:
        """Derive :math:`\rho v^2` with respect to the altitude.

        Args:
            altitude: The altitude.
            mach: The Mach number.

        Returns:
            The derivative of :math:`\rho v^2` with respect to the altitude.
        """
        drho_dh, dv_dh = self.__compute_drho_dh_dv_dh(mach, altitude)
        return (
            drho_dh * self.__velocity * self.__velocity
            + 2.0 * self.__rho * dv_dh * self.__velocity
        )

    def __compute_drhov2_dmach(
        self,
        altitude: float,
    ) -> float:
        """Derive :math:`\rho v^2` with respect to the Mach number.

        Args:
            altitude: The altitude.

        Returns:
            The derivative of :math:`\rho v^2` with respect to the Mach number.
        """
        dv_dmach = self.__compute_dv_dmach(altitude)
        return 2.0 * self.__rho * dv_dmach * self.__velocity

    def __compute_rho_v(
        self,
        mach: float,
        altitude: float,
    ) -> tuple[float, float]:
        """Compute the velocity and density from given Mach number and altitude.

        Args:
            mach: The Mach number.
            altitude: The altitude.

        Returns:
            The density, the velocity.
        """
        if altitude.real < 36089.0:
            self.__velocity = mach * 1116.39 * self.math.sqrt(1 - 6.875e-6 * altitude)
            self.__rho = 2.377e-3 * (1 - 6.875e-6 * altitude) ** 4.2561
        else:
            self.__velocity = mach * 968.1
            self.__rho = (
                2.377e-3 * 0.2971 * self.math.exp((36089.0 - altitude) / 20806.7)
            )

        return self.__rho, self.__velocity

    def __compute_dv_dmach(
        self,
        altitude: float,
    ) -> float:
        """Derive the velocity with respect to the Mach number from a given altitude.

        Args:
            altitude: The altitude.

        Returns:
            The derivative of the velocity with respect to the Mach number.
        """
        if altitude.real < 36089.0:
            return 1116.39 * self.math.sqrt(1 - 6.875e-6 * altitude)
        else:
            return 968.1

    def __compute_drho_dh_dv_dh(
        self,
        mach: float,
        altitude: float,
    ) -> tuple[float, float]:
        """Compute the derivative of the density and velocity wrt the altitude.

        Args:
            mach: The Mach number.
            altitude: The altitude.

        Returns:
            The derivative of the density with respect to the altitude,
            the derivative of the velocity with respect to the altitude.
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

    def __compute_adim_coeff(
        self,
        force: float,
        sref: float,
    ) -> float:
        """Compute the adimensional force coefficient from the force (lift or drag).

        Args:
            force: The force.
            sref: The reference surface.

        Returns:
            The adimensional force coefficient, either the drag or lift coefficient.
        """
        return force / (0.5 * self.__rhov2 * sref)

    def execute(
        self,
        x_shared: ndarray,
        y_12: ndarray,
        y_32: ndarray,
        x_2: ndarray,
        true_cstr: bool = False,
        c_4: ndarray | None = None,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        """Compute the drag and the lift-to-drag ratio.

        Args:
            x_shared: The values of the shared design variables,
                where ``x_shared[0]`` is the thickness/chord ratio,
                ``x_shared[1]`` is the altitude,
                ``x_shared[2]`` is the Mach number,
                ``x_shared[3]`` is the aspect ratio,
                ``x_shared[4]`` is the wing sweep and
                ``x_shared[5]`` is the wing surface area.
            y_12: The coupling variable from the structure disciplines,
                where ``y_12[0]`` is the total aircraft weight
                and ``y_12[1]`` is the wing twist.
            y_32: The coupling variable (engine scale factor)
                from the propulsion discipline,
            x_2: The friction coefficient.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            c_4: The minimum drag coefficient.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The aerodynamics outputs:
                - ``y_2``: The outputs of the aerodynamics analysis:
                    - ``y_2[0]``: the lift,
                    - ``y_2[1]``: the drag,
                    - ``y_2[2]``: the lift/drag ratio,
                - ``y_21``: The coupling variable (lift) for the structure discipline,
                - ``y_23``: The coupling variable (drag) for the propulsion discipline,
                - ``y_24``: The coupling variable (lift/drag ratio)
                   for the mission discipline,
                - ``g_2``: The pressure gradient to be constrained.
        """
        return self._execute(
            x_shared[0],
            x_shared[1],
            x_shared[2],
            x_shared[4],
            x_shared[5],
            y_12[0],
            y_12[1],
            y_32[0],
            x_2[0],
            true_cstr=true_cstr,
            c_4=c_4,
        )

    def _execute(
        self,
        tc_ratio: float,
        altitude: float,
        mach: float,
        sweep: float,
        wing_area: float,
        ac_mass: float,
        twist: float,
        esf: float,
        c_f: float,
        true_cstr: bool = False,
        c_4: ndarray | None = None,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        """Compute the drag and the lift-to-drag ratio.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            altitude: The altitude.
            mach: The Mach number.
            sweep: The wing sweep.
            wing_area: The wing surface area.
            ac_mass: The total aircraft weight.
            twist: The wing twist.
            esf: The engine scale factor.
            c_f: The friction coefficient.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            c_4: The minimum drag coefficient.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The aerodynamics outputs:
                - ``y_2``: The outputs of the aerodynamics analysis:
                    - ``y_2[0]``: the lift,
                    - ``y_2[1]``: the drag,
                    - ``y_2[2]``: the lift/drag ratio,
                - ``y_21``: The coupling variable (lift) for the structure discipline,
                - ``y_23``: The coupling variable (drag) for the propulsion discipline,
                - ``y_24``: The coupling variable (lift/drag ratio)
                   for the mission discipline,
                - ``g_2``: The pressure gradient to be constrained.
        """
        c_4 = c_4 or self.constants[4]

        y_2 = zeros(3, dtype=self.dtype)
        y_23 = zeros(1, dtype=self.dtype)
        y_24 = zeros(1, dtype=self.dtype)
        y_21 = zeros(1, dtype=self.dtype)
        g_2 = zeros(1, dtype=self.dtype)

        self.__compute_rho_v(mach, altitude)
        rhov2 = self.__compute_rhov2()
        lift_coeff = self.__compute_cl(wing_area, ac_mass)

        # Modification of CDmin for ESF and Cf
        fo1 = self.base.compute_polynomial_approximation(
            self.__esf_cf_initial,
            array([esf, c_f], dtype=self.dtype),
            self.__flag1,
            self.__bound1,
            self.__ao_coeff_1,
            self.__ai_coeff_1,
            self.__aij_coeff_1,
        )

        # Modification of drag_coeff for wing twist
        fo2 = self.base.compute_polynomial_approximation(
            self.__twist_initial,
            array([twist], dtype=self.dtype),
            self.__flag2,
            self.__bound2,
            self.__ao_coeff_2,
            self.__ai_coeff_2,
            self.__aij_coeff_2,
        )

        cdmin = self.__compute_cd_min(tc_ratio, sweep, fo1, c_4=c_4)
        self.__compute_k_aero(mach, sweep)
        drag_coeff = self.__compute_cd(lift_coeff, fo2, cdmin)
        drag = 0.5 * rhov2 * drag_coeff * wing_area

        y_2[1] = drag
        y_2[2] = lift_coeff / drag_coeff
        y_2[0] = ac_mass
        y_23[0] = y_2[1]
        y_24[0] = y_2[2]
        y_21[0] = y_2[0]

        # Computation of total drag of A/C
        # adverse pressure gradient
        g_2[0] = self.base.compute_polynomial_approximation(
            self.__tc_initial,
            array([tc_ratio], dtype=self.dtype),
            self.__flag3,
            self.__bound3,
            self.__ao_coeff_3,
            self.__ai_coeff_3,
            self.__aij_coeff_3,
        )
        # Custom Pgrad function: replace by a linear function
        # coeff_dir=(1.04-0.96)/(0.06-0.04)# = 4
        #         g_2[0] = coeff_dir*Z[0]+0.8
        if not true_cstr:
            g_2[0] -= self.PRESSURE_GRADIENT_LIMIT

        return y_2, y_21, y_23, y_24, g_2

    @staticmethod
    def __derive_liftoverdrag(
        cl_cd: ndarray,
        lift_jacobian: ndarray,
        drag_jacobian: ndarray,
        inv_drag: ndarray,
    ) -> ndarray:
        """Compute the Jacobian of the lift-over-drag ratio.

        Args:
            cl_cd: The lift-over-drag ratio.
            lift_jacobian: The Jacobian of the lift.
            drag_jacobian: The Jacobian of the drag.
            inv_drag: The inverse of the drag coefficient.

        Returns:
            The Jacobian of the lift-over-drag ratio.
        """
        return inv_drag * (lift_jacobian - cl_cd * drag_jacobian)

    @staticmethod
    def __set_coupling_jacobian(
        jacobian: dict[str, dict[str, ndarray]],
    ) -> dict[str, dict[str, ndarray]]:
        """Set the Jacobian sub-structure related to output coupling variables."""
        jacobian["y_21"]["x_2"] = jacobian["y_2"]["x_2"][0:1, :]
        jacobian["y_21"]["x_shared"] = jacobian["y_2"]["x_shared"][0:1, :]
        jacobian["y_21"]["y_12"] = jacobian["y_2"]["y_12"][0:1, :]
        jacobian["y_21"]["y_32"] = jacobian["y_2"]["y_32"][0:1, :]
        jacobian["y_21"]["c_4"] = jacobian["y_2"]["c_4"][0:1, :]

        jacobian["y_23"]["x_2"] = jacobian["y_2"]["x_2"][1:2, :]
        jacobian["y_23"]["x_shared"] = jacobian["y_2"]["x_shared"][1:2, :]
        jacobian["y_23"]["y_12"] = jacobian["y_2"]["y_12"][1:2, :]
        jacobian["y_23"]["y_32"] = jacobian["y_2"]["y_32"][1:2, :]
        jacobian["y_23"]["c_4"] = jacobian["y_2"]["c_4"][1:2, :]

        jacobian["y_24"]["x_2"] = jacobian["y_2"]["x_2"][2:3, :]
        jacobian["y_24"]["x_shared"] = jacobian["y_2"]["x_shared"][2:3, :]
        jacobian["y_24"]["y_12"] = jacobian["y_2"]["y_12"][2:3, :]
        jacobian["y_24"]["y_32"] = jacobian["y_2"]["y_32"][2:3, :]
        jacobian["y_24"]["c_4"] = jacobian["y_2"]["c_4"][2:3, :]
        return jacobian

    def __initialize_jacobian(self) -> dict[str, dict[str, ndarray]]:
        """Initialize the Jacobian structure.

        Returns:
            The Jacobian structure initialized with zeros.
        """
        jacobian = {"y_2": {}, "g_2": {}, "y_21": {}, "y_23": {}, "y_24": {}}

        jacobian["y_2"]["x_shared"] = zeros((3, 6), dtype=self.dtype)
        jacobian["y_2"]["x_2"] = zeros((3, 1), dtype=self.dtype)
        jacobian["y_2"]["y_12"] = zeros((3, 2), dtype=self.dtype)
        jacobian["y_2"]["y_32"] = zeros((3, 1), dtype=self.dtype)
        jacobian["y_2"]["c_4"] = zeros((3, 1), dtype=self.dtype)

        jacobian["g_2"]["x_2"] = zeros((1, 1), dtype=self.dtype)
        jacobian["g_2"]["x_shared"] = zeros((1, 6), dtype=self.dtype)
        jacobian["g_2"]["y_12"] = zeros((1, 2), dtype=self.dtype)
        jacobian["g_2"]["y_32"] = zeros((1, 1), dtype=self.dtype)
        jacobian["g_2"]["c_4"] = zeros((1, 1), dtype=self.dtype)
        return jacobian

    def linearize(
        self,
        x_shared: ndarray,
        y_12: ndarray,
        y_32: ndarray,
        x_2: ndarray,
        c_4: ndarray | None = None,
    ) -> dict[str, dict[str, ndarray]]:
        """Compute the Jacobian of the drag and lift-to-drag ratio.

        Args:
            x_shared: The values of the shared design variables,
                where ``x_shared[0]`` is the thickness/chord ratio,
                ``x_shared[1]`` is the altitude,
                ``x_shared[2]`` is the Mach number,
                ``x_shared[3]`` is the aspect ratio,
                ``x_shared[4]`` is the wing sweep and
                ``x_shared[5]`` is the wing surface area.
            y_12: The coupling variable from the structure disciplines,
                where ``y_12[0]`` is the total aircraft weight
                and ``y_12[1]`` is the wing twist.
            y_32: The coupling variable (engine scale factor)
                from the propulsion discipline,
            x_2: The friction coefficient.
            c_4: The minimum drag coefficient.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The Jacobian of the outputs.
        """
        return self._linearize(
            x_shared[0],
            x_shared[1],
            x_shared[2],
            x_shared[4],
            x_shared[5],
            y_12[0],
            y_12[1],
            y_32[0],
            x_2[0],
            c_4=c_4,
        )

    def _linearize(
        self,
        tc_ratio: float,
        altitude: float,
        mach: float,
        sweep: float,
        wing_area: float,
        ac_mass: float,
        twist: float,
        esf: float,
        c_f: float,
        c_4: float | None = None,
    ) -> dict[str, dict[str, ndarray]]:
        """Compute the Jacobian of the drag and lift-to-drag ratio.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            altitude: The altitude.
            mach: The Mach number.
            sweep: The wing sweep.
            wing_area: The wing surface area.
            ac_mass: The total aircraft weight.
            twist: The wing twist.
            esf: The engine scale factor.
            c_f: The friction coefficient.
            c_4: The minimum drag coefficient.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The Jacobian of the outputs.
        """
        jacobian = self.__initialize_jacobian()
        c_4 = c_4 or self.constants[4]
        self.__compute_rho_v(mach, altitude)
        rhov2 = self.__compute_rhov2()
        lift_coeff = ac_mass / (0.5 * rhov2 * wing_area)
        # Modification of drag_coeff_min for ESF and Cf
        (
            fo1,
            ai_coeff1,
            aij_coeff1,
            s_shifted1,
        ) = self.base.derive_polynomial_approximation(
            self.__esf_cf_initial,
            array([esf, c_f], dtype=self.dtype),
            self.__flag1,
            self.__bound1,
            self.__ao_coeff_1,
            self.__ai_coeff_1,
            self.__aij_coeff_1,
        )

        drag_coeff_incomp = c_4
        cdmin = self.__compute_cd_min(tc_ratio, sweep, fo1, c_4=c_4)

        # Modification of drag_coeff for wing twist
        (
            fo2,
            ai_coeff2,
            aij_coeff2,
            s_shifted2,
        ) = self.base.derive_polynomial_approximation(
            self.__twist_initial,
            array([twist], dtype=self.dtype),
            self.__flag2,
            self.__bound2,
            self.__ao_coeff_2,
            self.__ai_coeff_2,
            self.__aij_coeff_2,
        )

        k_aero = self.__compute_k_aero(mach, sweep)
        drag_coeff = fo2 * (cdmin + k_aero * lift_coeff * lift_coeff)
        cl_cd = lift_coeff / drag_coeff
        fo1fo2 = fo1 * fo2
        dy2dc4 = 0.5 * rhov2 * wing_area * fo1fo2
        jacobian["y_2"]["c_4"][1, :] = dy2dc4
        jacobian["y_2"]["c_4"][2, :] = -cl_cd / drag_coeff * fo1fo2
        cl2 = lift_coeff * lift_coeff
        drag = 0.5 * rhov2 * drag_coeff * wing_area
        dyn_pressure = 0.5 * rhov2
        dyn_force = dyn_pressure * wing_area
        inv_drag = 1.0 / drag

        # dDrag_dCf
        dadimcf_dcf = self.__compute_dadimcf_dcf(c_f)

        dy2dx2 = dyn_force * fo2 * drag_coeff_incomp * dadimcf_dcf
        dy2dx2 *= (
            ai_coeff1[1]
            + aij_coeff1[1, 0] * s_shifted1[0]
            + aij_coeff1[1, 1] * s_shifted1[1]
        )  # dDrag_dCf
        jacobian["y_2"]["x_2"][1, 0:1] = dy2dx2
        # d(Lift/Drag)_dCf

        dlod = self.__derive_liftoverdrag(
            cl_cd,
            jacobian["y_2"]["x_2"][0, 0],
            jacobian["y_2"]["x_2"][1, 0],
            inv_drag,
        )
        jacobian["y_2"]["x_2"][2, 0:1] = dlod

        # dDrag/d(t/c)
        dy2dxs = dyn_force * fo2 * 3.05 * 5.0 / 3.0 * tc_ratio ** (2.0 / 3.0)
        dy2dxs *= (self.math.cos(sweep * DEG_TO_RAD)) ** 1.5
        jacobian["y_2"]["x_shared"][1, 0] = dy2dxs

        # d(Drag)/dh
        drhov2_dh = self.__compute_drhov2_dh(altitude, mach)
        dcl_dh = self.__compute_dcl_dh(altitude, mach, wing_area, ac_mass)
        add_der = cl2 * drhov2_dh + rhov2 * 2.0 * lift_coeff * dcl_dh
        dy2dxs = (cdmin * drhov2_dh + k_aero * add_der) * 0.5 * wing_area * fo2
        jacobian["y_2"]["x_shared"][1, 1:2] = dy2dxs

        # d(Drag)/dM
        dcd_dmach = self.__compute_dcd_dmach(
            altitude, mach, sweep, wing_area, ac_mass, fo2
        )
        drhov2_dmach = self.__compute_drhov2_dmach(altitude)
        jacobian["y_2"]["x_shared"][1, 2:3] = (
            0.5 * wing_area * (drag_coeff * drhov2_dmach + rhov2 * dcd_dmach)
        )

        # d(Drag)/dsweep
        dcd_dsweep = self.__compute_dcd_dsweep(tc_ratio, mach, sweep, lift_coeff, fo2)
        jacobian["y_2"]["x_shared"][1, 4] = dyn_force * dcd_dsweep

        # d(Drag)/dsref
        dcd_dsef = self.__compute_dcd_dsref(wing_area, fo2)
        dy2dxs = drag / wing_area + dyn_force * dcd_dsef
        jacobian["y_2"]["x_shared"][1, 5:6] = dy2dxs

        for i in range(6):
            dy2dxs0i = jacobian["y_2"]["x_shared"][0, i]
            dy2dxs1i = jacobian["y_2"]["x_shared"][1, i]
            dlod = self.__derive_liftoverdrag(cl_cd, dy2dxs0i, dy2dxs1i, inv_drag)
            jacobian["y_2"]["x_shared"][2, i : i + 1] = dlod

        # d(Lift)/dWt
        jacobian["y_2"]["y_12"][0, 0] = 1.0

        # d(Drag)/dWt
        jacobian["y_2"]["y_12"][1, 0] = 2.0 * ac_mass * k_aero * fo2 / dyn_force

        # d(Drag)/dtwist
        dadimtwist_dadim = self.__compute_dadimtwist_dtwist(twist)
        jacobian["y_2"]["y_12"][1, 1:2] = (
            dyn_force
            * dadimtwist_dadim
            * (cdmin + k_aero * lift_coeff * lift_coeff)
            * (ai_coeff2[0] + aij_coeff2[0, 0] * s_shifted2[0])
        )
        # d(Lift/Drag)/dtwist
        for i in range(2):
            dy2dy120 = jacobian["y_2"]["y_12"][0, i]
            dy2dy121 = jacobian["y_2"]["y_12"][1, i]
            dy2dy122 = self.__derive_liftoverdrag(cl_cd, dy2dy120, dy2dy121, inv_drag)
            jacobian["y_2"]["y_12"][2, i : i + 1] = dy2dy122

        # dDrag/dESF
        dadimesf_desf = self.__compute_dadimesf_desf(esf)
        jacobian["y_2"]["y_32"][1, 0:1] = (
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
        jacobian["y_2"]["y_32"][2, 0:1] = self.__derive_liftoverdrag(
            cl_cd,
            jacobian["y_2"]["y_32"][0, 0],
            jacobian["y_2"]["y_32"][1, 0],
            inv_drag,
        )

        # d(dp/dx)/d(t/c)
        (
            _,
            ai_coeff3,
            aij_coeff3,
            s_shifted3,
        ) = self.base.derive_polynomial_approximation(
            self.__tc_initial,
            array([tc_ratio], dtype=self.dtype),
            self.__flag3,
            self.__bound3,
            self.__ao_coeff_3,
            self.__ai_coeff_3,
            self.__aij_coeff_3,
        )

        dadimtaper_dtaper = self.__compute_dadimtaper_dtaper(tc_ratio)
        jacobian["g_2"]["x_shared"][0, 0] = dadimtaper_dtaper * (
            ai_coeff3[0] + aij_coeff3[0, 0] * s_shifted3[0]
        )
        return self.__set_coupling_jacobian(jacobian)
