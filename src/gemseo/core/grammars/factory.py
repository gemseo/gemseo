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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory to instantiate a derived class of :class:`.BaseGrammar`."""
from __future__ import annotations

import logging
from typing import Any

from gemseo.core.factory import Factory
from gemseo.core.grammars.base_grammar import BaseGrammar

LOGGER = logging.getLogger(__name__)


class GrammarFactory:
    """A factory of :class:`.BaseGrammar`."""

    def __init__(self) -> None:  # noqa: D107
        self.__factory = Factory(BaseGrammar, ("gemseo.core.grammars",))

    def create(
        self,
        class_name: str,
        name: str,
        **options: Any,
    ) -> BaseGrammar:
        """Create a grammar.

        Args:
            class_name: The name of a class deriving from :class:`.BaseGrammar`.
            name: The name to be given to the grammar.
            **options: The options to be passed to the initialization.
        """
        return self.__factory.create(class_name, name=name, **options)

    @property
    def grammars(self) -> list[str]:
        """The sorted names of the available grammars."""
        return self.__factory.classes

    def is_available(self, class_name: str) -> bool:
        """Return whether a grammar class exists.

        Args:
            class_name: The name of a class deriving from :class:`.BaseGrammar`.

        Returns:
            Whether the grammar class exists.
        """
        return self.__factory.is_available(class_name)
