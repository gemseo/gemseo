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
from numpy import array
from numpy import hstack

if TYPE_CHECKING:
    from gemseo.typing import RealArray


@pytest.fixture(scope="module", params=[False, True])
def multioutput(request) -> bool:
    """The problem is multioutput."""
    return request.param


@pytest.fixture(scope="module")
def input_data() -> RealArray:
    """The input data."""
    return array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])


@pytest.fixture(scope="module")
def output_data(input_data, multioutput) -> RealArray:
    """The output data."""
    first_output = input_data[:, [0]] + input_data[:, [1]]
    if multioutput:
        return hstack((
            first_output,
            input_data[:, [0]] * input_data[:, [1]],
            input_data[:, [0]],
            input_data[:, [0]] ** 2,
        ))

    return first_output
