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
"""Continuous variable."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

from numpy import float64
from numpy import inf
from numpy import iscomplexobj
from numpy import logical_and

from gemseo.space._variable._base import BaseVariable
from gemseo.space._variable._base import ComponentDType
from gemseo.space._variable._base import DataType

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.util.typing import BooleanArray


class ContinuousVariable(BaseVariable):
    """A variable whose components are real numbers."""

    component_type: ClassVar[ComponentDType] = float64
    """The NumPy type of the components of the variable."""

    type: Literal[DataType.FLOAT] = DataType.FLOAT
    """The type of data."""

    def cast(self, value: ndarray) -> ndarray:
        """Cast a value of the variable to the NumPy type of the variable.

        The NumPy type of a complex value is left untouched
        so that the perturbation of a complex-step differentiation survives;
        such a value is copied,
        as a real one is by the cast,
        so that the caller does not keep a hand on it.

        Args:
            value: The value of the variable.

        Returns:
            The cast value of the variable.
        """
        return value.copy() if iscomplexobj(value) else super().cast(value)

    def compute_normalization_mask(  # noqa: D102
        self, enable_integer_normalization: bool
    ) -> BooleanArray:
        return logical_and(self.lower_bound != -inf, self.upper_bound != inf)
