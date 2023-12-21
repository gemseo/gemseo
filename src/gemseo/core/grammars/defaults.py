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
"""The grammar default values."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from pandas import DataFrame

from gemseo.core.discipline_data import DisciplineData
from gemseo.core.discipline_data import MutableData
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.core.grammars.base_grammar import BaseGrammar


class Defaults(DisciplineData):
    """A class for handling grammar default values.

    A dictionary-like interface to bind grammar names to default values. The namespace
    settings of the grammar are taken into account.
    """

    __grammar: BaseGrammar
    """The grammar bound to the defaults."""

    def __init__(
        self,
        grammar: BaseGrammar,
        data: MutableData,
    ) -> None:
        """
        Args:
            grammar: The grammar bound to the defaults.
        """  # noqa: D205, D212, D415
        super().__init__(input_to_namespaced=grammar.to_namespaced)
        self.__grammar = grammar
        # Explicitly set the items such that they are checked.
        self.update(data)

    def __setitem__(self, name: str, value: Any) -> None:
        if name not in self.__grammar:
            if isinstance(value, DataFrame):
                alien_names = {
                    f"{name}{self.SEPARATOR}{column}" for column in value.columns
                }.difference(self.__grammar.keys())
                if alien_names:
                    raise KeyError(
                        f"The names {pretty_str(alien_names)} "
                        "are not in the grammar."
                    )
            else:
                raise KeyError(f"The name {name} is not in the grammar.")
        super().__setitem__(name, value)

    def rename(self, name: str, new_name: str) -> None:
        """Rename a name.

        Args:
            name: The current name.
            new_name: The new name.
        """
        default_value = self.pop(name, None)
        if default_value is not None:
            self[new_name] = default_value
