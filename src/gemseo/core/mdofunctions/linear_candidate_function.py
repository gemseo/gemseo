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
"""Linear candidate function."""

from __future__ import annotations

from abc import abstractmethod

from gemseo.core.mdofunctions.mdo_function import MDOFunction


class LinearCandidateFunction(MDOFunction):
    """MDOFunction that may be linearized in a Scenario."""

    @property
    @abstractmethod
    def linear_candidate(self) -> bool:
        """Whether the final MDOFunction could be linear."""

    @property
    @abstractmethod
    def input_dimension(self) -> int | None:
        """The input variable dimension, needed for linear candidates.

        If ``None`` this cannot be determined nor by ``MDODiscipline`` default inputs
        nor by ``MDODisciplineAdapter.__input_names_to_sizes``.
        """
