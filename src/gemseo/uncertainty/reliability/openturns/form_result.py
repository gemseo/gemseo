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
"""Result of a FORM or SORM study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gemseo.uncertainty.reliability.result import ReliabilityResult

if TYPE_CHECKING:
    from gemseo.typing import RealArray


@dataclass(frozen=True)
class MPFP:
    """The design point, a.k.a. most probable failure point (MPFP)."""

    physical: RealArray
    """The design point in the physical space."""

    standard: RealArray
    """The design point in the standard space."""

    physical_as_dict: dict[str, RealArray]
    """The design point in the physical space split based on the variable names."""

    standard_as_dict: dict[str, RealArray]
    """The design point in the standard space split based on the variable names."""


@dataclass(frozen=True)
class ImportanceFactors:
    """The importance factors for a FORM or SORM study.

    The importance factors can be defined in three ways:

    - classical: the squares of the co-factors of the design point
      in the physical space,
    - elliptical: the squares of the co-factors of the design point
      in the standard space,
    - physical: the squares of the physical sensitivities,
      i.e. the partial derivatives of the Hasofer-Lind reliability index
      with respect to the inputs in the physical space.
    """

    classical: RealArray
    """The classical importance factors."""

    classical_as_dict: dict[str, RealArray]
    """The classical importance factors split based on the variable names."""

    elliptical: RealArray
    """The elliptical importance factors."""

    elliptical_as_dict: dict[str, RealArray]
    """The elliptical importance factors split based on the variable names."""

    physical: RealArray
    """The physical importance factors."""

    physical_as_dict: dict[str, RealArray]
    """The physical importance factors split based on the variable names."""


@dataclass(frozen=True)
class FORMResult(ReliabilityResult):
    """The result of a FORM or SORM study."""

    design_point: MPFP
    """The design point, a.k.a. most probable failure point (MPFP)."""

    importance_factors: ImportanceFactors
    """The importance factors."""

    reliability_index: float
    """The Hasofer-Lind reliability index."""
