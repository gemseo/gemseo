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

"""A factory of parameter spaces."""

from __future__ import annotations

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.base_factory import BaseFactory


class ParameterSpaceFactory(BaseFactory[ParameterSpace]):
    """A factory of parameter spaces."""

    _CLASS = ParameterSpace
    _PACKAGE_NAMES = ("gemseo.problems.uncertainty",)
