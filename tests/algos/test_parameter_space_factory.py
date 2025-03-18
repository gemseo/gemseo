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

import pytest

from gemseo.algos.parameter_space_factory import ParameterSpaceFactory


@pytest.fixture(scope="module")
def factory() -> ParameterSpaceFactory:
    """The factory of parameter spaces."""
    return ParameterSpaceFactory()


@pytest.mark.parametrize(
    ("class_name", "is_available"),
    [
        ("IshigamiSpace", True),
        ("WingWeightUncertainSpace", True),
        ("SellarDesignSpace", False),
    ],
)
def test_is_available(factory, class_name, is_available) -> None:
    """Check that the method is_available works."""
    assert factory.is_available(class_name) is is_available
