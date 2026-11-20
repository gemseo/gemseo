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
"""Tests for the base class of the encodings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.space import DesignSpace
from gemseo.space.encoding.base import BaseSpaceEncoding

if TYPE_CHECKING:
    from gemseo.util.typing import NumberArray


class _Encoding(BaseSpaceEncoding):
    """An encoding copying the original space."""

    n_builds: int
    """The number of times the encoded space has been built."""

    def __init__(self, space: DesignSpace) -> None:
        # Set before the base initializer, so that a build there is counted.
        self.n_builds = 0
        super().__init__(space)

    def _create_encoded_space(self) -> DesignSpace:  # noqa: D102
        self.n_builds += 1
        return DesignSpace(self._original_space.name)

    def encode(self, value: NumberArray) -> NumberArray:  # noqa: D102
        return value

    def decode(self, value: NumberArray) -> NumberArray:  # noqa: D102
        return value


def test_encoded_space_built_on_first_access_and_kept() -> None:
    """Check that the encoded space is built on the first access and then kept."""
    encoding = _Encoding(DesignSpace("space"))
    assert encoding.n_builds == 0

    encoded_space = encoding.encoded_space

    assert encoding.encoded_space is encoded_space
    assert encoding.n_builds == 1


def test_original_space() -> None:
    """Check that the original space is the space the encoding is built for."""
    space = DesignSpace("space")

    assert _Encoding(space).original_space is space


def test_create_constraints_default() -> None:
    """Check that an encoding creates no constraint by default."""
    assert _Encoding(DesignSpace()).create_constraints() == ()
