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
"""Base class for the derived-data collaborators of versioned variables."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.space.design._staleness_guard import StalenessGuard
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.space.design._variables import Variables


class RegistryDerivedData(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base managing staleness guards over data derived from versioned variables.

    Subclasses register one or more named guards, each with a callback that
    reconciles a piece of derived data with the registry (by rebuilding it
    eagerly or invalidating it for lazy rebuild). A guard fires only when the
    version key changed since its last refresh.
    """

    _variables: Variables
    """The variables."""

    __guards: dict[str, StalenessGuard]
    """The staleness guards keyed by name."""

    _DEFAULT_GUARD_NAME: ClassVar[str] = ""
    """The default guard name, for a single-guard subclass."""

    def __init__(self, variables: Variables) -> None:
        """
        Args:
            variables: The variables.
        """  # noqa: D205, D212
        self._variables = variables
        self.__guards = {}

    def _register_guard(
        self, rebuild: Callable[[], None], name: str = _DEFAULT_GUARD_NAME
    ) -> None:
        """Register a staleness guard reconciling a piece of derived data.

        Args:
            rebuild: The callback reconciling the derived data with the registry.
            name: The name identifying the guard.
                Leave empty for a single-guard subclass.
        """
        self.__guards[name] = StalenessGuard(rebuild)

    def _refresh(self, name: str = _DEFAULT_GUARD_NAME) -> None:
        """Reconcile the derived data of a guard if the version key changed.

        Args:
            name: The name of the guard.
                Leave empty for a single-guard subclass.
        """
        self.__guards[name].refresh(self._get_version_key())

    def _get_version_key(self) -> object:
        """Return the version key identifying the current registry state.

        Returns:
            The version key.
        """
        return self._variables.version
