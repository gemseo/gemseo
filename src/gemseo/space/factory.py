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

"""Factories of design and random spaces."""

from __future__ import annotations

from typing import ClassVar
from typing import Final

from gemseo.core.base_factory import BaseFactory
from gemseo.space.design import DesignSpace
from gemseo.space.random import RandomSpace


class DesignSpaceFactory(BaseFactory[DesignSpace]):
    """A factory of design spaces."""

    _class: ClassVar[type[DesignSpace]] = DesignSpace
    _package_names: ClassVar[tuple[str, ...]] = ("gemseo.problem",)


class RandomSpaceFactory(BaseFactory[RandomSpace]):
    """A factory of random spaces."""

    _class: ClassVar[type[RandomSpace]] = RandomSpace
    _package_names: ClassVar[tuple[str, ...]] = ("gemseo.problem.uncertainty",)


design_space_factory: Final[DesignSpaceFactory] = DesignSpaceFactory()
"""The factory for `DesignSpace` objects."""

random_space_factory: Final[RandomSpaceFactory] = RandomSpaceFactory()
"""The factory for `RandomSpace` objects."""
