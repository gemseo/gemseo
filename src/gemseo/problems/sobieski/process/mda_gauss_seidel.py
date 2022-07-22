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
from __future__ import annotations

from typing import Any

from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.problems.sobieski.disciplines import create_disciplines


class SobieskiMDAGaussSeidel(MDAGaussSeidel):
    """A Gauss-Seidel MDA for the Sobieski's SSBJ use case."""

    def __init__(
        self,
        dtype: str = "float64",
        **mda_options: Any,
    ) -> None:
        """
        Args:
            dtype: The NumPy type for data arrays, either "float64" or "complex128".
            **mda_options: The options of the MDA.
        """
        super().__init__(create_disciplines(dtype), **mda_options)
