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
"""Base class for displaying optimization data in the progress bar."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo import StrKeyMapping
    from gemseo.algos.hashable_ndarray import HashableNdarray
    from gemseo.algos.optimization_problem import OptimizationProblem


class BaseProgressBarData(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for the progress bar data related to an optimization problem."""

    _problem: OptimizationProblem
    """The optimization problem from which to retrieve the data."""

    def __init__(self, problem: OptimizationProblem) -> None:
        """
        Args:
            problem: The optimization problem from which to retrieve the data.
        """  # noqa: D205, D212
        self._problem = problem

    @abstractmethod
    def get(self, input_value: HashableNdarray | None) -> StrKeyMapping:
        """Return the data to be displayed in the progress bar.

        Args:
            input_value: The input value related to this data, if any.

        Returns:
            The data to be displayed in the progress bar.
        """
