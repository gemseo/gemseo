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
"""Read-only view over the variables of a random space."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.space.variable.random import RandomVariable
from gemseo.space.variables_view import VariablesView

if TYPE_CHECKING:
    from typing import Any

    from gemseo.space._random.variables import RandomVariables
    from gemseo.uncertainty.distribution.core.base_joint import BaseJointDistribution


class RandomVariablesView(VariablesView[RandomVariable]):
    """A read-only live view over a registry of random variables.

    In addition to the random variables,
    this view gives access to their dependency structure, namely their copulas,
    and to their joint probability distribution.

    Note:
        The joint probability distributions and the copulas
        wrap objects of a third-party library,
        which this view cannot freeze;
        they are read-only by contract.
    """

    __slots__ = ()

    _mapping: RandomVariables
    """The registry of the random variables."""

    @property
    def distribution(self) -> BaseJointDistribution | None:
        """The joint probability distribution of the random variables, if any."""
        return self._mapping.distribution

    @property
    def copulas(self) -> tuple[tuple[tuple[str, ...], Any], ...]:
        """The independent copulas defined by blocks of random variables."""
        return self._mapping.copulas
