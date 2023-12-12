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
"""Aerodynamics computation for the Sobieski's SSBJ use case."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemseo.problems.sobieski.core.utils import SobieskiBase

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import ndarray

LOGGER = logging.getLogger(__name__)


class SobieskiDiscipline:
    """Base class for the disciplines of the Sobieski's SSBJ use case."""

    # TODO: API: remove these unused class attributes.
    DTYPE_COMPLEX = SobieskiBase.DTYPE_COMPLEX
    DTYPE_DOUBLE = SobieskiBase.DTYPE_DOUBLE

    def __init__(self, sobieski_base: SobieskiBase) -> None:
        """
        Args:
            sobieski_base: The Sobieski base.
        """  # noqa: D205 D212
        self.base = sobieski_base
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

    @property
    def constants(self) -> ndarray:
        """The default constants."""
        return self.base.constants

    @property
    def dtype(self) -> SobieskiBase.DataType:
        """The NumPy data type."""
        return self.base.dtype

    @property
    def math(self) -> ModuleType:
        """The library of mathematical functions."""
        return self.base.math
