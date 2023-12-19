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
"""The full-factorial DOE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyDOE2
from numpy import array
from numpy import ndarray

from gemseo.algos.doe.base_full_factorial_doe import BaseFullFactorialDOE

if TYPE_CHECKING:
    from collections.abc import Sequence


class PyDOEFullFactorialDOE(BaseFullFactorialDOE):
    """The pyDOE based full-factorial DOE.

    .. note:: This class is a singleton.
    """

    def _generate_fullfact_from_levels(self, levels: int | Sequence[int]) -> ndarray:
        doe = pyDOE2.fullfact(levels)
        # Because pyDOE return the DOE where the values of levels are integers from 0 to
        # the maximum level number,
        # we need to divide by levels - 1.
        # To not divide by zero,
        # we first find the null denominators,
        # we replace them by one,
        # then we change the final values of the DOE by 0.5.
        divide_factor = array(levels) - 1
        null_indices = divide_factor == 0
        divide_factor[null_indices] = 1
        doe /= divide_factor
        doe[:, null_indices] = 0.5
        return doe
