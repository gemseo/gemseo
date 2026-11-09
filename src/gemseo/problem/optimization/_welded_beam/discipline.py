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
"""The discipline of the welded beam."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import NamedTuple

from numpy import array
from numpy import sqrt
from numpy import vstack

from gemseo.core.discipline import Discipline
from gemseo.problem.optimization._welded_beam import Welding
from gemseo.problem.optimization._welded_beam.data import beam_length
from gemseo.problem.optimization._welded_beam.data import get_properties
from gemseo.problem.optimization._welded_beam.data import initial_dimensions
from gemseo.problem.optimization._welded_beam.data import initial_material
from gemseo.problem.optimization._welded_beam.data import load
from gemseo.problem.optimization._welded_beam.data import max_deflection
from gemseo.problem.optimization._welded_beam.data import shear_stress_factor
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.util.typing import RealOrComplexArray
    from gemseo.util.typing import StrKeyMapping

    _Gradients = dict[str, dict[str, complex]]

_sqrt_2: Final[float] = sqrt(2.0)
"""The square root of 2."""

_buckling_factor: Final[float] = 4.013 / (6 * beam_length**2)
"""The factor of the buckling load."""

_weld_input_names: Final[tuple[str, str, str]] = (
    "weld_thickness",
    "weld_length",
    "bar_height",
)
"""The names of the inputs defining the geometry of the weld, in this order."""


class _Quantities(NamedTuple):
    """The inputs of the welded beam and the quantities derived from them."""

    h: complex
    """The weld thickness."""

    length: complex
    """The weld length."""

    t: complex
    """The bar height."""

    b: complex
    """The bar thickness."""

    c_1: complex
    """The unit cost of the weld."""

    c_2: complex
    """The unit cost of the bar."""

    s: complex
    """The allowable stress."""

    e: complex
    """The Young modulus."""

    g: complex
    """The shear modulus."""

    weld_volume: complex
    """The volume of the weld."""

    bar_volume: complex
    """The volume of the bar."""

    tau: complex
    """The shear stress in the weld."""

    allowable_shear_stress: complex
    """The allowable shear stress."""

    sigma: complex
    """The bending stress in the bar."""

    ratio: complex
    """The square root of the ratio of the Young modulus to the shear modulus."""

    critical_load: complex
    """The buckling load without the correction for the bar height."""

    buckling_load: complex
    """The buckling load of the bar."""

    delta: complex
    """The deflection of the beam."""


class WeldedBeam(Discipline):
    """The welded beam.

    This discipline computes the cost and the normalized constraints of the beam
    from its dimensions and from the properties of its material,
    which are inputs named after the properties of the
    [catalog][gemseo.problem.optimization._welded_beam.data.catalog].
    Its default inputs are the initial point of the design space,
    with the properties of the initial material
    instead of the catalog variable `material`,
    which needs a driver encoding the catalog variables.

    The Jacobian is analytic.
    With four-sided welding,
    the shear stress is not differentiable when the weld length equals the bar height;
    its derivatives there are the ones for a weld length smaller than the bar height.
    """

    _output_names: ClassVar[tuple[str, ...]] = (
        "cost",
        "shear_stress",
        "bending_stress",
        "thickness_difference",
        "buckling",
        "deflection",
    )
    """The names of the outputs."""

    __welding: Welding
    """The welding of the beam to its support."""

    __weld_height_factor: float
    """The factor $w$ of the bar height in the volume of the weld."""

    def __init__(self, welding: Welding | str = Welding.TWO_SIDED) -> None:
        """
        Args:
            welding: The welding of the beam to its support.

        Raises:
            ValueError: When the welding is not a value of
                [Welding][gemseo.problem.optimization._welded_beam.Welding].
        """  # noqa: D205 D212 D415
        try:
            welding = Welding(welding)
        except ValueError:
            msg = (
                f"{welding!r} is not a welding; "
                f"the weldings are {pretty_str(tuple(Welding), sort=False)}."
            )
            raise ValueError(msg) from None

        super().__init__()
        self.__welding = welding
        self.__weld_height_factor = 1.0 if welding is Welding.FOUR_SIDED else 0.0
        default_input_data = {
            name: array([value]) for name, value in initial_dimensions.items()
        } | get_properties(initial_material)
        self.io.input_grammar.update_from_names(default_input_data)
        self.io.output_grammar.update_from_names(self._output_names)
        self.io.input_grammar.defaults.update(default_input_data)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping:
        q = self.__compute_quantities(input_data)
        outputs = {
            "cost": (1 + q.c_1) * q.weld_volume + q.c_2 * q.bar_volume,
            "shear_stress": q.tau / q.allowable_shear_stress - 1,
            "bending_stress": q.sigma / q.s - 1,
            "thickness_difference": q.h - q.b,
            "buckling": 1 - q.buckling_load / load,
            "deflection": q.delta / max_deflection - 1,
        }
        return {name: array([value]) for name, value in outputs.items()}

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        gradients = self.__compute_gradients(
            self.__compute_quantities(self.io.input_data)
        )
        self.jac = {
            output_name: {
                input_name: array([[derivative]])
                for input_name, derivative in gradient.items()
            }
            for output_name, gradient in gradients.items()
        }
        self._init_jacobian(input_names, output_names, fill_missing_keys=True)

    def __compute_quantities(self, input_data: StrKeyMapping) -> _Quantities:
        """Compute the quantities from which the outputs are computed.

        Args:
            input_data: The input data.

        Returns:
            The inputs and the quantities derived from them.
        """
        h = input_data["weld_thickness"][0]
        length = input_data["weld_length"][0]
        t = input_data["bar_height"][0]
        b = input_data["bar_thickness"][0]
        s = input_data["allowable_stress"][0]
        e = input_data["young_modulus"][0]
        g = input_data["shear_modulus"][0]
        ratio = sqrt(e / g)
        critical_load = _buckling_factor * t * b**3 * sqrt(e * g)
        return _Quantities(
            h=h,
            length=length,
            t=t,
            b=b,
            c_1=input_data["weld_unit_cost"][0],
            c_2=input_data["bar_unit_cost"][0],
            s=s,
            e=e,
            g=g,
            weld_volume=(self.__weld_height_factor * t + length) * h**2,
            bar_volume=t * b * (beam_length + length),
            tau=self._compute_shear_stress(h, length, t, self.__welding),
            allowable_shear_stress=shear_stress_factor * s,
            sigma=6 * load * beam_length / (t**2 * b),
            ratio=ratio,
            critical_load=critical_load,
            buckling_load=critical_load * (1 - t * ratio / (4 * beam_length)),
            delta=4 * load * beam_length**3 / (e * t**3 * b),
        )

    def __compute_gradients(self, q: _Quantities) -> _Gradients:
        """Compute the gradients of the outputs.

        Args:
            q: The inputs and the quantities derived from them.

        Returns:
            The non-zero derivatives of the outputs, by input name.
        """
        h, length, t, b = q.h, q.length, q.t, q.b
        c_1, c_2, s, e, g = q.c_1, q.c_2, q.s, q.e, q.g
        w = self.__weld_height_factor
        d_tau = self._compute_shear_stress_gradient(h, length, t, q.tau, self.__welding)
        d_shear_stress = dict(
            zip(_weld_input_names, d_tau / q.allowable_shear_stress, strict=True)
        )
        d_shear_stress["allowable_stress"] = -q.tau / q.allowable_shear_stress / s
        d_ratio_term = q.critical_load * t * q.ratio / (8 * beam_length)
        d_buckling_load = {
            "bar_height": q.buckling_load / t
            - q.critical_load * q.ratio / (4 * beam_length),
            "bar_thickness": 3 * q.buckling_load / b,
            "young_modulus": (q.buckling_load / 2 - d_ratio_term) / e,
            "shear_modulus": (q.buckling_load / 2 + d_ratio_term) / g,
        }
        d_deflection = {
            "bar_height": -3 * q.delta / t,
            "bar_thickness": -q.delta / b,
            "young_modulus": -q.delta / e,
        }
        return {
            "cost": {
                "weld_thickness": 2 * (1 + c_1) * (w * t + length) * h,
                "weld_length": (1 + c_1) * h**2 + c_2 * t * b,
                "bar_height": (1 + c_1) * w * h**2 + c_2 * b * (beam_length + length),
                "bar_thickness": c_2 * t * (beam_length + length),
                "weld_unit_cost": q.weld_volume,
                "bar_unit_cost": q.bar_volume,
            },
            "shear_stress": d_shear_stress,
            "bending_stress": {
                "bar_height": -2 * q.sigma / (t * s),
                "bar_thickness": -q.sigma / (b * s),
                "allowable_stress": -q.sigma / s**2,
            },
            "thickness_difference": {"weld_thickness": 1.0, "bar_thickness": -1.0},
            "buckling": {
                name: -value / load for name, value in d_buckling_load.items()
            },
            "deflection": {
                name: value / max_deflection for name, value in d_deflection.items()
            },
        }

    @staticmethod
    def _compute_shear_stress(
        h: complex, length: complex, t: complex, welding: Welding
    ) -> complex:
        """Compute the shear stress in the weld.

        Args:
            h: The weld thickness.
            length: The weld length.
            t: The bar height.
            welding: The welding of the beam to its support.

        Returns:
            The shear stress in the weld.
        """
        area, inertia, radius, half_length = WeldedBeam._compute_weld_geometry(
            h, length, t, welding
        )
        primary = load / area
        secondary = load * (beam_length + length / 2) * radius / inertia
        cosine = half_length / radius
        return sqrt(primary**2 + secondary**2 + 2 * primary * secondary * cosine)

    @staticmethod
    def _compute_shear_stress_gradient(
        h: complex, length: complex, t: complex, tau: complex, welding: Welding
    ) -> RealOrComplexArray:
        """Compute the gradient of the shear stress in the weld.

        Args:
            h: The weld thickness.
            length: The weld length.
            t: The bar height.
            tau: The shear stress in the weld.
            welding: The welding of the beam to its support.

        Returns:
            The gradient of the shear stress
            with respect to the weld thickness, the weld length and the bar height.
        """
        area, inertia, radius, half_length = WeldedBeam._compute_weld_geometry(
            h, length, t, welding
        )
        d_area, d_inertia, d_radius, d_half_length = (
            WeldedBeam._compute_weld_geometry_jacobian(h, length, t, welding)
        )
        primary = load / area
        moment = load * (beam_length + length / 2)
        secondary = moment * radius / inertia
        cosine = half_length / radius
        d_primary = -primary * d_area / area
        d_moment = array([0.0, load / 2, 0.0])
        d_secondary = (
            d_moment * radius + moment * d_radius - secondary * d_inertia
        ) / inertia
        d_cosine = (d_half_length - cosine * d_radius) / radius
        return (
            primary * d_primary
            + secondary * d_secondary
            + cosine * (secondary * d_primary + primary * d_secondary)
            + primary * secondary * d_cosine
        ) / tau

    @staticmethod
    def _compute_weld_geometry(
        h: complex, length: complex, t: complex, welding: Welding
    ) -> RealOrComplexArray:
        r"""Compute the geometry of the weld.

        Args:
            h: The weld thickness.
            length: The weld length.
            t: The bar height.
            welding: The welding of the beam to its support.

        Returns:
            The area, the polar moment of inertia and the radius $R$ of the weld
            and the half-length $a$ such that $\cos\theta = a / R$.
        """
        if welding is Welding.FOUR_SIDED:
            return WeldedBeam._compute_four_sided_weld_geometry(h, length, t)

        return WeldedBeam._compute_two_sided_weld_geometry(h, length, t)

    @staticmethod
    def _compute_weld_geometry_jacobian(
        h: complex, length: complex, t: complex, welding: Welding
    ) -> RealOrComplexArray:
        """Compute the Jacobian of the geometry of the weld.

        Args:
            h: The weld thickness.
            length: The weld length.
            t: The bar height.
            welding: The welding of the beam to its support.

        Returns:
            The Jacobian of the geometry of the weld
            with respect to the weld thickness, the weld length and the bar height.
        """
        if welding is Welding.FOUR_SIDED:
            return WeldedBeam._compute_four_sided_weld_geometry_jacobian(h, length, t)

        return WeldedBeam._compute_two_sided_weld_geometry_jacobian(h, length, t)

    @staticmethod
    def _compute_two_sided_weld_geometry(
        h: complex, length: complex, t: complex
    ) -> RealOrComplexArray:
        r"""Compute the geometry of a two-sided weld.

        Args:
            h: The weld thickness.
            length: The weld length.
            t: The bar height.

        Returns:
            The area, the polar moment of inertia and the radius $R$ of the weld
            and the half-length $a$ such that $\cos\theta = a / R$.
        """
        area = _sqrt_2 * h * length
        factor = (h + t) ** 2 / 4 + length**2 / 12
        radius = sqrt(length**2 + (h + t) ** 2) / 2
        return array([area, area * factor, radius, length / 2])

    @staticmethod
    def _compute_two_sided_weld_geometry_jacobian(
        h: complex, length: complex, t: complex
    ) -> RealOrComplexArray:
        """Compute the Jacobian of the geometry of a two-sided weld.

        Args:
            h: The weld thickness.
            length: The weld length.
            t: The bar height.

        Returns:
            The Jacobian of the geometry of the weld
            with respect to the weld thickness, the weld length and the bar height.
        """
        area = _sqrt_2 * h * length
        factor = (h + t) ** 2 / 4 + length**2 / 12
        radius = sqrt(length**2 + (h + t) ** 2) / 2
        d_area = _sqrt_2 * array([length, h, 0.0])
        d_factor = array([(h + t) / 2, length / 6, (h + t) / 2])
        return vstack((
            d_area,
            d_area * factor + area * d_factor,
            array([h + t, length, h + t]) / (4 * radius),
            array([0.0, 0.5, 0.0]),
        ))

    @staticmethod
    def _compute_four_sided_weld_geometry(
        h: complex, length: complex, t: complex
    ) -> RealOrComplexArray:
        r"""Compute the geometry of a four-sided weld.

        It is made of a two-sided weld and of a two-sided weld
        whose length and height are swapped:
        their areas and polar moments of inertia add up,
        the radius is the largest one
        and the half-length is the one of the weld with this radius.

        Args:
            h: The weld thickness.
            length: The weld length.
            t: The bar height.

        Returns:
            The area, the polar moment of inertia and the radius $R$ of the weld
            and the half-length $a$ such that $\cos\theta = a / R$.
        """
        values = WeldedBeam._compute_two_sided_weld_geometry(h, length, t)
        swapped_values = WeldedBeam._compute_two_sided_weld_geometry(h, t, length)
        radial_values = (
            swapped_values[2:]
            if WeldedBeam._is_swapped_radius(values, swapped_values)
            else values[2:]
        )
        return array([*(values[:2] + swapped_values[:2]), *radial_values])

    @staticmethod
    def _compute_four_sided_weld_geometry_jacobian(
        h: complex, length: complex, t: complex
    ) -> RealOrComplexArray:
        """Compute the Jacobian of the geometry of a four-sided weld.

        Args:
            h: The weld thickness.
            length: The weld length.
            t: The bar height.

        Returns:
            The Jacobian of the geometry of the weld
            with respect to the weld thickness, the weld length and the bar height.
        """
        is_swapped_radius = WeldedBeam._is_swapped_radius(
            WeldedBeam._compute_two_sided_weld_geometry(h, length, t),
            WeldedBeam._compute_two_sided_weld_geometry(h, t, length),
        )
        jacobian = WeldedBeam._compute_two_sided_weld_geometry_jacobian(h, length, t)
        swapped_jacobian = WeldedBeam._compute_two_sided_weld_geometry_jacobian(
            h, t, length
        )[:, [0, 2, 1]]
        radial_jacobian = swapped_jacobian[2:] if is_swapped_radius else jacobian[2:]
        return vstack((jacobian[:2] + swapped_jacobian[:2], radial_jacobian))

    @staticmethod
    def _is_swapped_radius(
        values: RealOrComplexArray, swapped_values: RealOrComplexArray
    ) -> bool:
        """Check whether the radius of a four-sided weld is the swapped one.

        Args:
            values: The geometry of the two-sided weld.
            swapped_values: The geometry of the two-sided weld
                whose length and height are swapped.

        Returns:
            Whether the radius of the swapped weld is the largest one.
        """
        # The real parts are compared so that the complex step can be used.
        return bool(swapped_values[2].real > values[2].real)
