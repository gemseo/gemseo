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
"""An MDA using the Jacobi algorithm for the Sobieski's SSBJ use case."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.mda.jacobi import MDAJacobi
from gemseo.problems.mdo.sobieski.core.utils import SobieskiBase
from gemseo.problems.mdo.sobieski.disciplines import create_disciplines

if TYPE_CHECKING:
    from gemseo.mda.jacobi_settings import MDAJacobi_Settings


class SobieskiMDAJacobi(MDAJacobi):
    """An [MDAJacobi][gemseo.mda.jacobi.MDAJacobi] for the Sobieski's SSBJ use case."""

    def __init__(
        self,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
        settings: MDAJacobi_Settings | None = None,
    ) -> None:
        """
        Args:
            dtype: The NumPy type for data arrays, either "float64" or "complex128".
        """  # noqa: D205 D212
        super().__init__(create_disciplines(dtype), settings=settings)
