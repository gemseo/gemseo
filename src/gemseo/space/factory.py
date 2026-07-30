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

"""Factories of design and parameter spaces."""

from __future__ import annotations

from typing import Final

from gemseo.core.base_factory import BaseFactory
from gemseo.space.design import DesignSpace
from gemseo.space.parameter import ParameterSpace


class DesignSpaceFactory(BaseFactory[DesignSpace]):
    """A factory of design spaces."""

    _CLASS = DesignSpace
    _PACKAGE_NAMES = ("gemseo.problem",)


class ParameterSpaceFactory(BaseFactory[ParameterSpace]):
    """A factory of parameter spaces."""

    _CLASS = ParameterSpace
    _PACKAGE_NAMES = ("gemseo.problem.uncertainty",)


DESIGN_SPACE_FACTORY: Final[DesignSpaceFactory] = DesignSpaceFactory()
"""The factory for `DesignSpace` objects."""

PARAMETER_SPACE_FACTORY: Final[ParameterSpaceFactory] = ParameterSpaceFactory()
"""The factory for `ParameterSpace` objects."""
