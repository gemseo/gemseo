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
"""Structure discipline for the Sobieski's SSBJ use case."""
from __future__ import annotations

import logging

from numpy import append
from numpy import array
from numpy import ndarray
from numpy import ones
from numpy import zeros

from gemseo.problems.sobieski.core.discipline import SobieskiDiscipline
from gemseo.problems.sobieski.core.utils import DEG_TO_RAD
from gemseo.problems.sobieski.core.utils import SobieskiBase

LOGGER = logging.getLogger(__name__)


class SobieskiStructure(SobieskiDiscipline):
    """Structure discipline for the Sobieski's SSBJ use case."""

    STRESS_LIMIT = 1.09
    TWIST_UPPER_LIMIT = 1.04
    TWIST_LOWER_LIMIT = 0.8

    def __init__(self, sobieski_base: SobieskiBase) -> None:
        super().__init__(sobieski_base)
        self.__ao_coeff_secthick = zeros(1, dtype=self.dtype)
        self.__ai_coeff_secthick = zeros(1, dtype=self.dtype)
        self.__aij_coeff_secthick = zeros((1, 1), dtype=self.dtype)
        self.__ao_coeff_twist = zeros(1, dtype=self.dtype)
        self.__ai_coeff_twist = zeros(4, dtype=self.dtype)
        self.__aij_coeff_twist = zeros((4, 4), dtype=self.dtype)
        self.__ao_coeff_stress = zeros(1, dtype=self.dtype)
        self.__ai_coeff_stress = zeros(5, dtype=self.dtype)
        self.__aij_coeff_stress = zeros((5, 5), dtype=self.dtype)
        self.__flag1 = array([2, 4, 4, 3], dtype=self.dtype)
        self.__bound1 = array([0.25, 0.25, 0.25, 0.25], dtype=self.dtype)
        self.__flag_stress = array([4, 1, 4, 1, 1], dtype=self.dtype)
        self.__flag_secthick = array([1], dtype=self.dtype)
        self.__bound_secthick = array([0.008], dtype=self.dtype)
        self.__s_initial_for_wing_twist = array(
            [
                self.x_initial,
                self.half_span_initial,
                self.aero_center_initial,
                self.lift_initial,
            ],
            dtype=self.dtype,
        )
        self.__s_initial_for_wing_weight = array([self.x_initial], dtype=self.dtype)
        self.__s_initial_for_constraints = array(
            [
                self.tc_initial,
                self.lift_initial,
                self.x_initial,
                self.half_span_initial,
                self.aero_center_initial,
            ],
            dtype=self.dtype,
        )

        self.__loc_ones = ones(5, dtype=self.dtype)
        self.__ones_mat = ones(5, dtype=self.dtype)
        self.__aero_center = None
        self.__half_span = None
        self.__dadimlift_dlift = None

    def __compute_dadimcenter_dcenter(self) -> float:
        """Derive the adimensioned aerodynamic center wrt the aerodynamic center.

        Returns:
            The derivative of the adimensioned aerodynamic center
            wrt the aerodynamic center.
        """
        return self.base.derive_normalization(
            self.aero_center_initial, self.__aero_center
        )

    def __compute_dcenter_dlambda(self, wing_taper_ratio: float) -> float:
        """Derive the aerodynamic center with respect to the wing taper ratio.

        Args:
            wing_taper_ratio: The wing taper ratio.

        Returns:
            The derivative of the aerodynamic center
            with respect to the wing taper ratio.
        """
        dadimcenter_dcenter = self.__compute_dadimcenter_dcenter()
        return dadimcenter_dcenter * 1.0 / (3.0 * (1.0 + wing_taper_ratio) ** 2)

    def __compute_wing_weight(
        self,
        tc_ratio: float,
        aspect_ratio: float,
        sweep: float,
        wing_area: float,
        wing_taper_ratio: float,
        wingbox_area: float,
        lift: float,
        linearize: bool = False,
        c_2: float | None = None,
    ) -> float:
        """Compute the weight of the wing.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            aspect_ratio: The aspect ratio.
            sweep: The wing sweep.
            wing_area: The wing surface area.
            wing_taper_ratio: The wing taper ratio.
            wingbox_area: The wingbox x-sectional area.
            lift: The lift coefficient.
            linearize: Whether to derive the polynomial approximation.
            c_2: The maximum load factor.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            If ``linearize`` is ``True``,
            the wing weight, the wing weight coefficient
            and the value of the polynomial function.
            Otherwise, the wing weight only.
        """
        c_2 = c_2 or self.constants[2]

        s_new = array([wingbox_area], dtype=self.dtype)
        if linearize:
            f_o, a_i, a_ij, s_shifted = self.base.derive_polynomial_approximation(
                self.__s_initial_for_wing_weight,
                s_new,
                self.__flag_secthick,
                self.__bound_secthick,
                self.__ao_coeff_secthick,
                self.__ai_coeff_secthick,
                self.__aij_coeff_secthick,
            )
        else:
            f_o = self.base.compute_polynomial_approximation(
                self.__s_initial_for_wing_weight,
                s_new,
                self.__flag_secthick,
                self.__bound_secthick,
                self.__ao_coeff_secthick,
                self.__ai_coeff_secthick,
                self.__aij_coeff_secthick,
            )
        wing_weight_coeff = (
            0.0051
            * ((lift * c_2) ** 0.557)
            * (wing_area**0.649)
            * (aspect_ratio**0.5)
            * (tc_ratio**-0.4)
            * ((1 + wing_taper_ratio) ** 0.1)
            * ((self.math.cos(sweep * DEG_TO_RAD)) ** -1.0)
            * ((0.1875 * wing_area) ** 0.1)
        )
        wing_weight = wing_weight_coeff * f_o
        if linearize:
            return wing_weight, wing_weight_coeff, f_o, a_i, a_ij, s_shifted

        return wing_weight

    def __compute_fuelwing_weight(
        self,
        tc_ratio: float,
        aspect_ratio: float,
        wing_area: float,
    ) -> float:
        """Compute the fuel wing weight.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            aspect_ratio: The aspect ratio.
            wing_area: The wing surface area.

        Returns:
            The fuel wing weight.
        """
        thickness = self.base.compute_thickness(aspect_ratio, tc_ratio, wing_area)
        return (5.0 * wing_area / 18.0) * (2.0 / 3.0 * thickness) * 42.5

    def __compute_dfuelwing_dtoverc(
        self,
        aspect_ratio: float,
        wing_area: float,
    ) -> float:
        """Derive the wing fuel weight.

        Args:
            aspect_ratio: The aspect ratio.
            wing_area: The wing area.

        Returns:
            The derivative of the wing fuel weight.
        """
        return 212.5 / 27.0 * wing_area ** (3.0 / 2.0) / self.math.sqrt(aspect_ratio)

    @staticmethod
    def __compute_dfuelwing_dar(
        tc_ratio: float,
        aspect_ratio: float,
        wing_area: float,
    ) -> float:
        """Derive the wing fuel weight with respect to the aspect ratio.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            aspect_ratio: The aspect ratio.
            wing_area: The wing surface.

        Returns:
            The derivative of the wing fuel weight with respect to the aspect ratio.
        """
        dfuelwing_dar = 212.5 / 27.0 * wing_area ** (3.0 / 2.0)
        dfuelwing_dar *= tc_ratio * -0.5 * aspect_ratio ** (-3.0 / 2.0)
        return dfuelwing_dar

    def __compute_dfuelwing_dsref(
        self,
        tc_ratio: float,
        aspect_ratio: float,
        wing_area: float,
    ) -> float:
        """Derive the wing fuel weight with respect to the reference surface.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            aspect_ratio: The aspect ratio.
            wing_area: The wing surface area.

        Returns:
            The derivative of the fuel wing weight with respect to
            the reference surface.
        """
        return 637.5 / 54.0 * wing_area**0.5 * tc_ratio / self.math.sqrt(aspect_ratio)

    def __compute_dhalfspan_dar(self, wing_area: float) -> float:
        """Derive the half-span with respect to the aspect ratio.

        Args:
            wing_area: The wing surface area.

        Returns:
            The derivative of the half-span with respect to the aspect ratio.
        """
        dadimspan_dspan = self.__compute_dadimspan_dspan()
        return dadimspan_dspan * wing_area / (8.0 * self.__half_span)

    def __compute_dadimspan_dsref(self, aspect_ratio: float) -> float:
        """Derive the half-span with respect to the reference surface.

        Args:
            aspect_ratio: The aspect ratio.

        Returns:
            The derivative of the half-span with respect to the
            reference surface.
        """
        dadimspan_dspan = self.__compute_dadimspan_dspan()
        return dadimspan_dspan * aspect_ratio / (8.0 * self.__half_span)

    def __compute_dadimspan_dspan(self) -> float:
        """Derive the adimensioned half-span with respect to the half-span.

        Returns:
            The derivative of the adimensioned half-span with respect to the half-span.
        """
        return self.base.derive_normalization(self.half_span_initial, self.__half_span)

    def __compute_dadimx_dx(self, wingbox_sectional_area: float) -> float:
        """Derive the adimensional sectional area with respect to the sectional area.

        Args:
            wingbox_sectional_area: The wingbox x-sectional area.

        Returns:
            The derivative of the adimensional sectional area
            with respect to the sectional area.
        """
        return self.base.derive_normalization(self.x_initial, wingbox_sectional_area)

    def __compute_dadimtaper_dtaper(self, tc_ratio: float) -> float:
        """Derive the adimensional taper ratio with respect to the taper ratio.

        Args:
            tc_ratio: The thickness-to-chord ratio.

        Returns:
            The derivative of the adimensional taper ratio
            with respect to the taper ratio.
        """
        return self.base.derive_normalization(self.tc_initial, tc_ratio)

    def __compute_dadimlift_dlift(self, lift: float) -> float:
        """Derive the adimensioned lift with respect to the lift.

        Args:
            lift: The lift coefficient.

        Returns:
            The derivative of the adimensioned lift with respect to the lift.
        """
        return self.base.derive_normalization(self.lift_initial, lift)

    @staticmethod
    def __compute_weight_ratio(
        w_t: float,
        w_f: float,
    ) -> float:
        """Computation the weight ratio from the Breguet formula.

        Args:
            w_t: The total aircraft mass.
            w_f: The fuel mass.

        Returns:
            The weight ratio.
        """
        return w_t / (w_t - w_f)

    def execute(
        self,
        x_shared: ndarray,
        y_21: ndarray,
        y_31: ndarray,
        x_1: ndarray,
        true_cstr: bool = False,
        c_0: ndarray | None = None,
        c_1: ndarray | None = None,
        c_2: ndarray | None = None,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        """Compute the structural outputs and the structural constraints.

        Args:
            x_shared: The values of the shared design variables,
                where ``x_shared[0]`` is the thickness-to-chord ratio,
                ``x_shared[1]`` is the altitude,
                ``x_shared[2]`` is the Mach number,
                ``x_shared[3]`` is the aspect ratio,
                ``x_shared[4]`` is the wing sweep and
                ``x_shared[5]`` is the wing surface area.
            y_21: The lift coefficient.
            y_31: The engine weight.
            x_1: The wing taper ratio ``x_1[0]``
                and the wingbox x-sectional area ``x_1[1]``.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            c_0: The minimum fuel weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.
            c_1: The miscellaneous weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.
            c_2: The maximum load factor.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The structural outputs and the structural constraints.
        """
        return self._execute(
            x_shared[0],
            x_shared[3],
            x_shared[4],
            x_shared[5],
            x_1[0],
            x_1[1],
            y_21[0],
            y_31[0],
            true_cstr=true_cstr,
            c_0=c_0,
            c_1=c_1,
            c_2=c_2,
        )

    def _execute(
        self,
        tc_ratio: float,
        aspect_ratio: float,
        sweep: float,
        wing_area: float,
        taper_ratio: float,
        wingbox_area: float,
        lift: float,
        engine_mass: float,
        true_cstr: bool = False,
        c_0: ndarray | None = None,
        c_1: ndarray | None = None,
        c_2: ndarray | None = None,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        """Compute the structural outputs and the structural constraints.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            aspect_ratio: The aspect ratio.
            sweep: The wing sweep.
            wing_area: The wing surface area.
            taper_ratio: The wing taper ratio.
            wingbox_area: The wingbox x-sectional area.
            lift: The lift coefficient.
            engine_mass: The mass of the engine.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            c_0: The minimum fuel weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.
            c_1: The miscellaneous weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.
            c_2: The maximum load factor.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The structural outputs and the structural constraints.
        """
        c_0 = c_0 or self.constants[0]
        c_1 = c_1 or self.constants[1]
        c_2 = c_2 or self.constants[2]
        y_12 = zeros(2, dtype=self.dtype)
        y_14 = zeros(2, dtype=self.dtype)

        self.__aero_center = self.base.compute_aero_center(taper_ratio)
        self.__half_span = self.base.compute_half_span(aspect_ratio, wing_area)

        y_1, y_11 = self.__poly_structure(
            tc_ratio,
            aspect_ratio,
            sweep,
            wing_area,
            taper_ratio,
            wingbox_area,
            lift,
            engine_mass,
            c_0=c_0,
            c_1=c_1,
            c_2=c_2,
        )

        g_1 = self.__poly_structure_constraints(tc_ratio, wingbox_area, lift)
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

    def __initialize_jacobian(self) -> dict[str, dict[str, ndarray]]:
        """Initialize the Jacobian structure.

        Returns:
            The empty Jacobian structure.
        """
        # Jacobian matrix as a dictionary
        jacobian = {"y_1": {}, "g_1": {}, "y_12": {}, "y_14": {}, "y_11": {}}

        n_y1 = 3
        jacobian["y_1"]["x_shared"] = zeros((n_y1, 6), dtype=self.dtype)
        jacobian["y_1"]["x_1"] = zeros((n_y1, 2), dtype=self.dtype)
        jacobian["y_1"]["y_21"] = zeros((n_y1, 1), dtype=self.dtype)
        jacobian["y_1"]["y_31"] = zeros((n_y1, 1), dtype=self.dtype)
        jacobian["y_1"]["c_0"] = zeros((n_y1, 1), dtype=self.dtype)
        jacobian["y_1"]["c_1"] = zeros((n_y1, 1), dtype=self.dtype)
        jacobian["y_1"]["c_2"] = zeros((n_y1, 1), dtype=self.dtype)

        n_y11 = 1
        jacobian["y_11"]["x_shared"] = zeros((n_y11, 6), dtype=self.dtype)
        jacobian["y_11"]["x_1"] = zeros((n_y11, 2), dtype=self.dtype)
        jacobian["y_11"]["y_21"] = zeros((n_y11, 1), dtype=self.dtype)
        jacobian["y_11"]["y_31"] = zeros((n_y11, 1), dtype=self.dtype)
        jacobian["y_11"]["c_0"] = zeros((n_y11, 1), dtype=self.dtype)
        jacobian["y_11"]["c_1"] = zeros((n_y11, 1), dtype=self.dtype)
        jacobian["y_11"]["c_2"] = zeros((n_y11, 1), dtype=self.dtype)

        n_y12 = 2
        jacobian["y_12"]["x_shared"] = zeros((n_y12, 6), dtype=self.dtype)
        jacobian["y_12"]["x_1"] = zeros((n_y12, 2), dtype=self.dtype)
        jacobian["y_12"]["y_21"] = zeros((n_y12, 1), dtype=self.dtype)
        jacobian["y_12"]["y_31"] = zeros((n_y12, 1), dtype=self.dtype)
        jacobian["y_12"]["c_0"] = zeros((n_y12, 1), dtype=self.dtype)
        jacobian["y_12"]["c_1"] = zeros((n_y12, 1), dtype=self.dtype)
        jacobian["y_12"]["c_2"] = zeros((n_y12, 1), dtype=self.dtype)

        for key, jac_val in jacobian["y_12"].items():
            jacobian["y_14"][key] = jac_val

        return jacobian

    def linearize(
        self,
        x_shared: ndarray,
        y_21: ndarray,
        y_31: ndarray,
        x_1: ndarray,
        true_cstr: bool = False,
        c_0: float | None = None,
        c_1: float | None = None,
        c_2: float | None = None,
    ) -> dict[str, dict[str, ndarray]]:
        """Derive the structural outputs and the structural constraints.

        Args:
            x_shared: The values of the shared design variables,
                where ``x_shared[0]`` is the thickness-to-chord ratio,
                ``x_shared[1]`` is the altitude,
                ``x_shared[2]`` is the Mach number,
                ``x_shared[3]`` is the aspect ratio,
                ``x_shared[4]`` is the wing sweep and
                ``x_shared[5]`` is the wing surface area.
            y_21: The lift coefficient.
            y_31: The engine weight.
            x_1: The wing taper ratio ``x_1[0]``
                and the wingbox x-sectional area ``x_1[1]``.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            c_0: The minimum fuel weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.
            c_1: The miscellaneous weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.
            c_2: The maximum load factor.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The derivatives of the structural outputs and the structural constraints.
        """
        return self._linearize(
            x_shared[0],
            x_shared[3],
            x_shared[4],
            x_shared[5],
            x_1[0],
            x_1[1],
            y_21[0],
            y_31[0],
            true_cstr=true_cstr,
            c_0=c_0,
            c_1=c_1,
            c_2=c_2,
        )

    def _linearize(
        self,
        tc_ratio: float,
        aspect_ratio: float,
        sweep: float,
        wing_area: float,
        taper_ratio: float,
        wingbox_area: float,
        lift: float,
        engine_mass: float,
        true_cstr: bool = False,
        c_0: float | None = None,
        c_1: float | None = None,
        c_2: float | None = None,
    ) -> dict[str, dict[str, ndarray]]:
        """Derive the structural outputs and the structural constraints.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            aspect_ratio: The aspect ratio.
            sweep: The wing sweep.
            wing_area: The wing surface area.
            taper_ratio: The wing taper ratio.
            wingbox_area: The wingbox x-sectional area.
            lift: The lift coefficient.
            engine_mass: The mass of the engine.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.
            c_0: The minimum fuel weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.
            c_1: The miscellaneous weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.
            c_2: The maximum load factor.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The derivatives of the structural outputs and the structural constraints.
        """

        c_0 = c_0 or self.constants[0]
        c_1 = c_1 or self.constants[1]
        c_2 = c_2 or self.constants[2]
        self.__aero_center = self.base.compute_aero_center(taper_ratio)
        self.__half_span = self.base.compute_half_span(aspect_ratio, wing_area)
        self.__dadimlift_dlift = self.__compute_dadimlift_dlift(lift)

        # Jacobian matrix as a dictionary
        jacobian = self.__initialize_jacobian()

        jacobian = self.__derive_poly_structure(
            jacobian,
            tc_ratio,
            aspect_ratio,
            sweep,
            wing_area,
            taper_ratio,
            wingbox_area,
            lift,
            engine_mass,
            c_2=c_2,
        )

        y_1, y_11 = self.__poly_structure(
            tc_ratio,
            aspect_ratio,
            sweep,
            wing_area,
            taper_ratio,
            wingbox_area,
            lift,
            engine_mass,
            c_0=c_0,
            c_1=c_1,
            c_2=c_2,
        )
        f = y_1[0] / (y_1[0] - y_1[1])
        jacobian["y_1"]["c_0"][0, 0] = 1
        jacobian["y_1"]["c_0"][1, 0] = 1
        fp = (1 * (y_1[0] - y_1[1]) - y_1[0] * (1 - 1)) / (y_1[0] - y_1[1]) ** 2
        jacobian["y_11"]["c_0"][:, 0] = fp / f

        jacobian["y_1"]["c_1"][0, 0] = 1
        fp = (1 * (y_1[0] - y_1[1]) - y_1[0] * (1 - 0)) / (y_1[0] - y_1[1]) ** 2
        jacobian["y_11"]["c_1"][:, 0] = fp / f

        dy10dc2 = (
            self.__compute_wing_weight(
                tc_ratio,
                aspect_ratio,
                sweep,
                wing_area,
                taper_ratio,
                wingbox_area,
                lift,
                c_2=c_2,
            )
            * 0.557
            / c_2
        )
        jacobian["y_1"]["c_2"][0, 0] = dy10dc2
        dy11dc2 = jacobian["y_1"]["c_2"][1, 0]
        fp = (dy10dc2 * (y_1[0] - y_1[1]) - y_1[0] * (dy10dc2 - dy11dc2)) / (
            y_1[0] - y_1[1]
        ) ** 2
        jacobian["y_11"]["c_2"][:, 0] = fp / f

        # Stress constraints
        jacobian = self.__derive_constraints(
            jacobian,
            tc_ratio,
            aspect_ratio,
            wing_area,
            taper_ratio,
            wingbox_area,
            lift,
            true_cstr=true_cstr,
        )

        # Twist constraints
        jacobian["g_1"]["x_1"][5, :] = jacobian["y_1"]["x_1"][2, :]
        jacobian["g_1"]["x_shared"][5, :] = jacobian["y_1"]["x_shared"][2, :]
        jacobian["g_1"]["y_21"][5, :] = jacobian["y_1"]["y_21"][2, :]
        jacobian["g_1"]["y_31"][5, :] = jacobian["y_1"]["y_31"][2, :]
        jacobian["g_1"]["c_0"][5, :] = jacobian["y_1"]["c_0"][2, :]
        jacobian["g_1"]["c_1"][5, :] = jacobian["y_1"]["c_1"][2, :]
        jacobian["g_1"]["c_2"][5, :] = jacobian["y_1"]["c_2"][2, :]

        if not true_cstr:
            jacobian["g_1"]["x_1"][6, :] = -jacobian["y_1"]["x_1"][2, :]
            jacobian["g_1"]["x_shared"][6, :] = -jacobian["y_1"]["x_shared"][2, :]
            jacobian["g_1"]["y_21"][6, :] = -jacobian["y_1"]["y_21"][2, :]
            jacobian["g_1"]["y_31"][6, :] = -jacobian["y_1"]["y_31"][2, :]
            jacobian["g_1"]["c_0"][6, :] = -jacobian["y_1"]["c_2"][2, :]
            jacobian["g_1"]["c_1"][6, :] = -jacobian["y_1"]["c_2"][2, :]
            jacobian["g_1"]["c_2"][6, :] = -jacobian["y_1"]["c_2"][2, :]

        # Coupling variables
        jacobian = self.__set_coupling_jacobian(jacobian)

        return jacobian

    @staticmethod
    def __set_coupling_jacobian(
        jacobian,
    ) -> dict[str, dict[str, ndarray]]:
        """Set Jacobian of the coupling variables."""
        for der_v, jac_loc in jacobian["y_1"].items():
            jacobian["y_12"][der_v][1, :] = jac_loc[2, :]
            jacobian["y_12"][der_v][0, :] = jac_loc[0, :]
            jacobian["y_14"][der_v] = jac_loc[0:2, :]

        return jacobian

    def __derive_poly_structure(
        self,
        jacobian: dict[str, dict[str, ndarray]],
        tc_ratio: float,
        aspect_ratio: float,
        sweep: float,
        wing_area: float,
        taper_ratio: float,
        wingbox_area: float,
        lift: float,
        engine_mass: float,
        c_2: float | None = None,
    ):
        """Derive the structural variables.

        Args:
            jacobian: The Jacobian of the discipline.
            tc_ratio: The thickness-to-chord ratio.
            aspect_ratio: The aspect ratio.
            sweep: The wing sweep.
            wing_area: The wing surface area.
            taper_ratio: The wing taper ratio.
            wingbox_area: The wingbox x-sectional area.
            lift: The lift coefficient.
            c_2: The maximum load factor.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The updated Jacobian of the discipline.
        """
        c_2 = c_2 or self.constants[2]
        # wing aero. center location
        y_1, _ = self.__poly_structure(
            tc_ratio,
            aspect_ratio,
            sweep,
            wing_area,
            taper_ratio,
            wingbox_area,
            lift,
            engine_mass,
        )

        # dWf/d(t/c)
        jacobian["y_1"]["x_shared"][1, 0] = self.__compute_dfuelwing_dtoverc(
            aspect_ratio, wing_area
        )

        # dWf/d(AR)
        jacobian["y_1"]["x_shared"][1, 3] = self.__compute_dfuelwing_dar(
            tc_ratio, aspect_ratio, wing_area
        )

        # dWf/dsref
        jacobian["y_1"]["x_shared"][1, 5] = self.__compute_dfuelwing_dsref(
            tc_ratio, aspect_ratio, wing_area
        )

        # Derivation of wing twist = y_1[2]
        s_1 = array(
            [wingbox_area, self.__half_span, self.__aero_center, lift],
            dtype=self.dtype,
        )
        _, a_i, a_ij, s_shifted = self.base.derive_polynomial_approximation(
            self.__s_initial_for_wing_twist,
            s_1,
            self.__flag1,
            self.__bound1,
            self.__ao_coeff_twist,
            self.__ai_coeff_twist,
            self.__aij_coeff_twist,
        )

        dcenter_dlambda = self.__compute_dcenter_dlambda(taper_ratio)
        jacobian["y_1"]["x_1"][2, 0] = dcenter_dlambda * (
            a_i[2]
            + a_ij[2, 0] * s_shifted[0]
            + a_ij[2, 1] * s_shifted[1]
            + a_ij[2, 2] * s_shifted[2]
            + a_ij[2, 3] * s_shifted[3]
        )

        dadimx_dx = self.__compute_dadimx_dx(wingbox_area)
        jacobian["y_1"]["x_1"][2, 1] = dadimx_dx * (
            a_i[0]
            + a_ij[0, 0] * s_shifted[0]
            + a_ij[0, 1] * s_shifted[1]
            + a_ij[0, 2] * s_shifted[2]
            + a_ij[0, 3] * s_shifted[3]
        )

        dhalfspan_dar = self.__compute_dhalfspan_dar(wing_area)
        jacobian["y_1"]["x_shared"][2, 3] = dhalfspan_dar * (
            a_i[1]
            + a_ij[1, 0] * s_shifted[0]
            + a_ij[1, 1] * s_shifted[1]
            + a_ij[1, 2] * s_shifted[2]
            + a_ij[1, 3] * s_shifted[3]
        )

        dhalfspan_dsref = self.__compute_dadimspan_dsref(aspect_ratio)
        jacobian["y_1"]["x_shared"][2, 5] = dhalfspan_dsref * (
            a_i[1]
            + a_ij[1, 0] * s_shifted[0]
            + a_ij[1, 1] * s_shifted[1]
            + a_ij[1, 2] * s_shifted[2]
            + a_ij[1, 3] * s_shifted[3]
        )

        jacobian["y_1"]["y_21"][2, 0] = self.__dadimlift_dlift * (
            a_i[3]
            + a_ij[0, 3] * s_shifted[0]
            + a_ij[1, 3] * s_shifted[1]
            + a_ij[2, 3] * s_shifted[2]
        )

        # Derivation of total weight = y_1[0] (requires derivation of wing weight)
        # Calculation of wingbox X-sectional thickness
        (
            wing_w,
            ww_coeff,
            _,
            a_i,
            a_ij,
            s_shifted,
        ) = self.__compute_wing_weight(
            tc_ratio,
            aspect_ratio,
            sweep,
            wing_area,
            taper_ratio,
            wingbox_area,
            lift,
            linearize=True,
            c_2=c_2,
        )

        # dtotal_weight_d(t/c)
        jacobian["y_1"]["x_shared"][0, 0] = (
            -0.4 * tc_ratio ** (-1.0) * wing_w + jacobian["y_1"]["x_shared"][1, 0]
        )

        dy11_dz = jacobian["y_1"]["x_shared"]
        # dtotal_weight_dAR
        dy11_dz[0, 3] = 0.5 * aspect_ratio ** (-1.0) * wing_w
        dy11_dz[0, 3] += jacobian["y_1"]["x_shared"][1, 3]

        # d1total_weight_dsweep
        dy11_dz[0, 4] = DEG_TO_RAD
        dy11_dz[0, 4] *= self.math.sin(sweep * DEG_TO_RAD) * wing_w
        dy11_dz[0, 4] /= self.math.cos(sweep * DEG_TO_RAD)

        # dtotal_weight_dsref
        dy11_dz[0, 5] = 0.749 * wing_area ** (-1.0) * wing_w
        dy11_dz[0, 5] += jacobian["y_1"]["x_shared"][1, 5]

        # dtotal_weight_d(lambda)
        jacobian["y_1"]["x_1"][0, 0] = (
            0.1 * (1 + taper_ratio) ** (-1.0) * wing_w + jacobian["y_1"]["x_1"][1, 0]
        )

        # dtotal_weight_dx
        jacobian["y_1"]["x_1"][0, 1] = ww_coeff * dadimx_dx
        jacobian["y_1"]["x_1"][0, 1] *= a_i[0] + a_ij[0, 0] * s_shifted[0]

        jacobian["y_1"]["x_1"][0, 1] += jacobian["y_1"]["x_1"][1, 1]

        # dtotal_weight_dLift
        jacobian["y_1"]["y_21"][0, 0] = 0.557 * lift**-1 * wing_w
        # dtotal_weight_dWe
        jacobian["y_1"]["y_31"][0, 0] = 1.0

        for out_v, jac_y_1 in jacobian["y_1"].items():
            val = jac_y_1[0, :] / y_1[0]
            val -= (jac_y_1[0, :] - jac_y_1[1, :]) / (y_1[0] - y_1[1])
            jacobian["y_11"][out_v][0, :] = val

        return jacobian

    def __derive_constraints(
        self,
        jacobian: dict[str, dict[str, ndarray]],
        tc_ratio: float,
        aspect_ratio: float,
        wing_area: float,
        taper_ratio: float,
        wingbox_area: float,
        lift: float,
        true_cstr: bool = False,
    ):
        """Derive the structural constraints from a polynomial approximation.

        Args:
            jacobian: The Jacobian of the discipline.
            tc_ratio: The thickness-to-chord ratio.
            aspect_ratio: The aspect ratio.
            wing_area: The wing surface area.
            taper_ratio: The wing taper ratio.
            wingbox_area: The wingbox x-sectional area.
            lift: The lift coefficient.
            true_cstr: If ``True``,
                return the value of the constraint outputs.
                Otherwise,
                return the distance to the corresponding constraint thresholds.

        Returns:
            The structural constraints from a polynomial approximation.
        """
        if true_cstr is False:
            n_g = 7
        else:
            n_g = 6
        jacobian["g_1"]["x_shared"] = zeros((n_g, 6), dtype=self.dtype)
        jacobian["g_1"]["x_1"] = zeros((n_g, 2), dtype=self.dtype)
        jacobian["g_1"]["y_21"] = zeros((n_g, 1), dtype=self.dtype)
        jacobian["g_1"]["y_31"] = zeros((n_g, 1), dtype=self.dtype)
        jacobian["g_1"]["c_0"] = zeros((n_g, 1), dtype=self.dtype)
        jacobian["g_1"]["c_1"] = zeros((n_g, 1), dtype=self.dtype)
        jacobian["g_1"]["c_2"] = zeros((n_g, 1), dtype=self.dtype)

        # Stress constraints
        s_new = array(
            [tc_ratio, lift, wingbox_area, self.__half_span, self.__aero_center],
            dtype=self.dtype,
        )

        for i in range(5):
            _, a_i, a_ij, s_shifted = self.base.derive_polynomial_approximation(
                self.__s_initial_for_constraints,
                s_new,
                self.__flag_stress,
                0.1 * self.__ones_mat + i * 0.05 * self.__ones_mat,
                self.__ao_coeff_stress,
                self.__ai_coeff_stress,
                self.__aij_coeff_stress,
            )
            dg_dx_1, dg_dz, dg_dy_21, dg_dy_31 = self.__der_constraint(
                taper_ratio,
                wingbox_area,
                tc_ratio,
                aspect_ratio,
                wing_area,
                a_i,
                a_ij,
                s_shifted,
            )
            jacobian["g_1"]["x_1"][i, :] = dg_dx_1[:]
            jacobian["g_1"]["x_shared"][i, :] = dg_dz[:]
            jacobian["g_1"]["y_21"][i, :] = dg_dy_21[:]
            jacobian["g_1"]["y_31"][i, :] = dg_dy_31[:]

        return jacobian

    def __poly_structure(
        self,
        tc_ratio: float,
        aspect_ratio: float,
        sweep: float,
        wing_area: float,
        taper_ratio: float,
        wingbox_area: float,
        lift: float,
        engine_mass: float,
        c_0: float | None = None,
        c_1: float | None = None,
        c_2: float | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Compute the structural variables from a polynomial approximation.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            aspect_ratio: The aspect ratio.
            sweep: The wing sweep.
            wing_area: The wing surface area.
            taper_ratio: The wing taper ratio.
            wingbox_area: The wingbox x-sectional area.
            lift: The lift coefficient.
            engine_mass: The mass of the engine.
            c_0: The minimum fuel weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.
            c_1: The miscellaneous weight.
                If ``None``, use :meth:`.SobieskiBase.constants`.
            c_2: The maximum load factor.
                If ``None``, use :meth:`.SobieskiBase.constants`.

        Returns:
            The vector of the total aircraft mass, fuel mass and wing twist,
            as well as the mass term in the Breguet range equation.
        """
        c_0 = c_0 or self.constants[0]
        c_1 = c_1 or self.constants[1]
        c_2 = c_2 or self.constants[2]
        y_1 = zeros(3, dtype=self.dtype)
        #         t = self.base.compute_thickness(x_shared)  # wing thickness

        # Calculation of wing twist
        y_1[2] = self.__compute_wing_twist(wingbox_area, lift)

        # Calculation of wingbox X-sectional thickness
        wing_w = self.__compute_wing_weight(
            tc_ratio,
            aspect_ratio,
            sweep,
            wing_area,
            taper_ratio,
            wingbox_area,
            lift,
            c_2=c_2,
        )
        fuel_wing_weight = self.__compute_fuelwing_weight(
            tc_ratio, aspect_ratio, wing_area
        )
        y_1_i1 = c_0 + fuel_wing_weight
        y_1[1] = y_1_i1  # Fuel weight
        y_1[0] = c_1 + wing_w + y_1_i1 + engine_mass

        # This is the mass term in the Breguet range equation.
        y_11 = array(
            [self.math.log(self.__compute_weight_ratio(y_1[0], y_1[1]))],
            dtype=self.dtype,
        )

        return y_1, y_11

    def __compute_wing_twist(
        self,
        wingbox_area: float,
        lift: float,
    ) -> ndarray:
        """Compute the wing twist.

        Args:
            wingbox_area: The wingbox x-sectional area.
            lift: The lift coefficient.

        Returns:
            The wing twist.
        """
        # Compute the half span and the aerodynamic center
        s_1 = array(
            [wingbox_area, self.__half_span, self.__aero_center, lift],
            dtype=self.dtype,
        )
        return self.base.compute_polynomial_approximation(
            self.__s_initial_for_wing_twist,
            s_1,
            self.__flag1,
            self.__bound1,
            self.__ao_coeff_twist,
            self.__ai_coeff_twist,
            self.__aij_coeff_twist,
        )

    def __der_constraint(
        self,
        taper_ratio: float,
        wingbox_area: float,
        tc_ratio: float,
        aspect_ratio: float,
        wing_area: float,
        a_i: ndarray,
        a_ij: ndarray,
        s_shifted: ndarray,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """Derive the structural constraints.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            aspect_ratio: The aspect ratio.
            wing_area: The wing surface area.
            a_i: The linear coefficients.
            a_ij: The quadratic coefficients.
            s_shifted: The normalized design variables.

        Returns:
            The derivatives of the structural constraints.
        """
        dcenter_dlambda = self.__compute_dcenter_dlambda(taper_ratio)

        dadimx_dx = self.__compute_dadimx_dx(wingbox_area)

        dadimhalfspan_dar = self.__compute_dhalfspan_dar(wing_area)

        dadimhalfspan_dsref = self.__compute_dadimspan_dsref(aspect_ratio)

        dadimtaper_dtaper = self.__compute_dadimtaper_dtaper(tc_ratio)

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
        dg_dy_21[0, 0] = self.__dadimlift_dlift * (
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

    def __poly_structure_constraints(
        self,
        tc_ratio: float,
        wingbox_area: float,
        lift: float,
    ) -> ndarray:
        """Compute the structural constraints from a polynomial approximation.

        Args:
            tc_ratio: The thickness-to-chord ratio.
            wingbox_area: The wingbox x-sectional area.
            lift: The lift coefficient.

        Returns:
            The structural constraints.
        """
        g_1 = zeros(6, dtype=self.dtype)
        s_new = array(
            [tc_ratio, lift, wingbox_area, self.__half_span, self.__aero_center],
            dtype=self.dtype,
        )
        for i in range(5):
            g_1[i] = self.base.compute_polynomial_approximation(
                self.__s_initial_for_constraints,
                s_new,
                self.__flag_stress,
                0.1 * self.__loc_ones + i * 0.05 * self.__loc_ones,
                self.__ao_coeff_stress,
                self.__ai_coeff_stress,
                self.__aij_coeff_stress,
            )

        return g_1
