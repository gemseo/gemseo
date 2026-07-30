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
"""Factory of reliability analysis algorithms."""

from __future__ import annotations

from typing import Final

from gemseo.core.base_factory import BaseFactory
from gemseo.uncertainty.reliability.core.base import BaseReliabilityAlgorithm


class ReliabilityAlgorithmFactory(BaseFactory):
    """The factory of reliability analysis algorithms."""

    # TODO: subclass BaseAlgorithmFactory once the MR 2434 has been merged.

    _CLASS = BaseReliabilityAlgorithm
    _PACKAGE_NAMES = ("gemseo.uncertainty.reliability",)


RELIABILITY_ALGORITHM_FACTORY: Final[ReliabilityAlgorithmFactory] = (
    ReliabilityAlgorithmFactory()
)
"""The factory of reliability analysis algorithms."""
