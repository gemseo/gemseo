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
"""Fixtures shared by the tests of the variable hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.space.variable.utils import ALL_KINDS
from tests.space.variable.utils import KIND_TO_KWARGS

if TYPE_CHECKING:
    from gemseo.space.variable import BaseVariable


@pytest.fixture(params=ALL_KINDS)
def variable(request) -> BaseVariable:
    """A variable of each kind, of size 1 and with the bounds [0, 1]."""
    return request.param(**KIND_TO_KWARGS[request.param])
