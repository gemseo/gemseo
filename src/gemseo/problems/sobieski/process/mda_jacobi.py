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

from typing import Any

from gemseo.mda.jacobi import MDAJacobi
from gemseo.problems.sobieski.core.utils import SobieskiBase
from gemseo.problems.sobieski.disciplines import create_disciplines


class SobieskiMDAJacobi(MDAJacobi):
    """An :class:`.MDAJacobi` for the Sobieski's SSBJ use case."""

    def __init__(
        self,
        n_processes: int = 1,
        dtype: SobieskiBase.DataType = SobieskiBase.DataType.FLOAT,
        **mda_options: Any,
    ) -> None:
        """
        Args:
            n_processes: The maximum simultaneous number of threads,
                if ``use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
            dtype: The NumPy type for data arrays, either "float64" or "complex128".
            **mda_options: The options of the MDA.
        """  # noqa: D205 D212
        super().__init__(
            create_disciplines(dtype), n_processes=n_processes, **mda_options
        )
