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

from numpy import array
from numpy import full
from numpy import ndarray
from openturns import Box

from gemseo.algos.doe.base_full_factorial_doe import BaseFullFactorialDOE

if TYPE_CHECKING:
    from collections.abc import Iterable


class OTFullFactorialDOE(BaseFullFactorialDOE):
    """The full-factorial DOE.

    .. note:: This class is a singleton.
    """

    def _generate_fullfact_from_levels(self, levels: Iterable[int]) -> ndarray:
        # This method relies on openturns.Box.
        # This latter assumes that the levels provided correspond to the intermediate
        # levels between lower and upper bounds, while GEMSEO includes these bounds
        # in the definition of the levels, so we subtract 2 in order to get
        # only intermediate levels.
        levels = [level - 2 for level in levels]

        # If any level is negative, we take them out, generate the DOE,
        # then append the DOE with 0.5 for the missing levels.
        ot_indices = []
        ot_levels = []
        for ot_index, ot_level in enumerate(levels):
            if ot_level >= 0:
                ot_levels.append(ot_level)
                ot_indices.append(ot_index)

        if not ot_levels:
            return full([1, len(levels)], 0.5)

        ot_doe = array(Box(ot_levels).generate())

        if len(ot_levels) == len(levels):
            return ot_doe

        doe = full([ot_doe.shape[0], len(levels)], 0.5)
        doe[:, ot_indices] = ot_doe
        return doe
