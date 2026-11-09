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
"""The data of the welded beam."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

from numpy import array

from gemseo.space.catalog.catalog import Catalog

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.util.typing import RealArray

load: Final[float] = 6000.0
"""The load $F$ applied at the free end of the beam, in lb."""

beam_length: Final[float] = 14.0
"""The length $L$ of the beam, in inches."""

max_deflection: Final[float] = 0.25
r"""The maximum deflection $\delta_{\max}$ of the beam, in inches."""

shear_stress_factor: Final[float] = 0.577
"""The ratio of the allowable shear stress to the allowable stress."""

step: Final[float] = 0.0625
"""The step between two consecutive values of a discrete dimension, in inches."""

catalog: Final[Catalog] = Catalog(
    properties={
        "weld_unit_cost": (0.1047, 0.0489, 0.5235, 0.5584),
        "bar_unit_cost": (0.0481, 0.0224, 0.2405, 0.2566),
        "allowable_stress": (3e4, 8e3, 5e3, 8e3),
        "young_modulus": (3e7, 14e6, 1e7, 16e6),
        "shear_modulus": (12e6, 6e6, 4e6, 6e6),
    },
    labels=("steel", "cast_iron", "aluminum", "brass"),
)
"""The catalog of the materials of the welded beam.

Its properties are
the unit cost $c_1$ of the weld,
the unit cost $c_2$ of the bar,
the allowable stress $S$ in psi,
the Young modulus $E$ in psi
and the shear modulus $G$ in psi.
"""

initial_material: Final[str] = "steel"
"""The material at the initial point."""

initial_dimensions: Final[Mapping[str, float]] = MappingProxyType({
    "weld_thickness": 0.5,
    "weld_length": 5.0,
    "bar_height": 10.0,
    "bar_thickness": 0.5,
})
"""The dimensions at the initial point, feasible with steel whatever the welding."""


def get_properties(label: str) -> dict[str, RealArray]:
    """Return the properties of a material as input data.

    Args:
        label: The label of the material.

    Returns:
        The properties of the material, as arrays of size 1.
    """
    position = catalog.get_position(label)
    return {
        name: array([values[position]]) for name, values in catalog.properties.items()
    }
