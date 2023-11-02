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

from typing import TYPE_CHECKING

import pytest
from numpy import arange
from numpy import int64 as np_int

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(scope="module")
def data() -> NDArray[np_int]:
    """A data array with 10 entries and 3 features.

    The values are:
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
            [24, 25, 26],
            [27, 28, 29],
        ]
    """
    return arange(30).reshape(10, 3)


@pytest.fixture(scope="module")
def small_data() -> NDArray[np_int]:
    """A data array with 3 entries and 3 features.

    The values are:     [         [0, 1, 2],         [3, 4, 5],         [6, 7, 8],     ]
    """
    return arange(9).reshape(3, 3)
