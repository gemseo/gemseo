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
"""Helpers of the tests of the space transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.space.transformation._working import get_default_transformation_factories
from gemseo.space.transformation.composition import SpaceComposition

if TYPE_CHECKING:
    from gemseo.space.base import BaseVariableSpace


def create_default_transformation(
    space: BaseVariableSpace,
    normalize: bool = False,
    relax_integer: bool = False,
    relax_discrete: bool = False,
) -> SpaceComposition:
    """Create the composition of the working transformations a driver asks for.

    Args:
        space: The space the problem is defined on.
        normalize: Whether the working values are normalized.
        relax_integer: Whether the integer variables are relaxed to float ones.
        relax_discrete: Whether the discrete variables are relaxed to float ones.

    Returns:
        The composition, possibly empty.
    """
    return SpaceComposition(
        space,
        *get_default_transformation_factories(
            space,
            normalize=normalize,
            relax_integer=relax_integer,
            relax_discrete=relax_discrete,
        ),
    )
