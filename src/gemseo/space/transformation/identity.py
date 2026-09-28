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

"""The neutral element of a composition of transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.space.transformation.base import BaseSpaceTransformation
from gemseo.space.transformation.base import _SpaceT

if TYPE_CHECKING:
    from gemseo.util.typing import NumberArray


class SpaceIdentity(BaseSpaceTransformation[_SpaceT]):
    """A transformation leaving the space and its values unchanged.

    Its working space is its original space, the very same object.
    A composition drops it at construction, so it costs nothing at evaluation time.
    """

    is_affine: ClassVar[bool] = True

    def _create_working_space(self) -> _SpaceT:  # noqa: D102
        return self._original_space

    def transform_value(self, value: NumberArray) -> NumberArray:  # noqa: D102
        return value

    def inverse_transform_value(  # noqa: D102
        self, value: NumberArray, no_check: bool = False
    ) -> NumberArray:
        return value

    def transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return jacobian

    def inverse_transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return jacobian
