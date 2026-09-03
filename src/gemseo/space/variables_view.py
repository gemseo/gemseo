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
"""Read-only view over the variables of a space."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.space._variable import BaseVariable
from gemseo.util.read_only_mapping import ReadOnlyMapping

if TYPE_CHECKING:
    # The only registry of variables is that of a design space for the time being;
    # this view uses nothing specific to a design space.
    from gemseo.space.design._variables import Variables


class VariablesView(ReadOnlyMapping[str, BaseVariable]):
    """A read-only live view over a registry of variables.

    A space of variables exposes its registry through such a view,
    so that the variables can only be mutated by the methods of the space,
    which maintain the data derived from the registry,
    e.g. the current value of a design space.

    The registry is held by reference,
    so a mutation of the space is reflected through the view;
    the view itself forbids insertion, deletion and update,
    and the variables it gives access to are immutable.
    """

    __slots__ = ()

    _mapping: Variables
    """The registry of the variables."""

    @property
    def name_to_indices(self) -> ReadOnlyMapping[str, range]:
        """The map from a variable name to an index range in the full vector."""
        return self._mapping.name_to_indices

    @property
    def has_integer_variables(self) -> bool:
        """Whether at least one variable is of integer type."""
        return self._mapping.has_integer_variables
