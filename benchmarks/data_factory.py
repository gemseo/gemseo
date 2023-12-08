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
"""Benchmark data producer base class."""

from __future__ import annotations

from dataclasses import dataclass
from pprint import pformat
from typing import Any
from typing import ClassVar

from numpy import zeros


@dataclass
class DataFactory:
    """Create a data structure as a dictionary."""

    with_only_ndarrays: ClassVar[bool] = False
    """Whether to only use data with NumPy array values."""

    items_nb: int = 1
    """The number of items in containers items."""

    keys_nb: int = 1
    """The number of keys for dictionaries."""

    depth: int = 0
    """The depth level for the nested dictionaries."""

    @property
    def data(self) -> dict[str, Any]:
        """The data structure."""
        data = {}
        self.__fill_dict(-1, self.__create_atom_data(), data)
        return data

    def __create_atom_data(self) -> dict[str, Any]:
        """Create an atomic data structure.

        Returns:
            The atomic data structure.
        """
        data_atom_ = {
            "ndarray": zeros((self.items_nb,)),
        }

        if not self.with_only_ndarrays:
            data_atom_.update({
                "float": 0.0,
                "int": 0,
                "bool": True,
                "string": "0" * self.items_nb,
                "list": [0] * self.items_nb,
            })
            # data_atom_["list_list"] = [data_atom_["list"]] * self.items_nb
            # data_atom_["dict_list"] = [data_atom_.copy()] * self.items_nb
            # data_atom_["dataframe"] = DataFrame(data_atom_["ndarray"])

        data_atom = {}
        for key_id in range(self.keys_nb):
            for key, value in data_atom_.items():
                data_atom[f"{key}_{key_id}"] = value

        return data_atom

    def __fill_dict(
        self, depth: int, data_atom: dict[str, Any], data: dict[str, Any]
    ) -> None:
        """Fill the data structure recursively.

        Args:
            depth: The current depth level.
            data_atom: The atomic data structure to be duplicated.
            data: The resulting data structure.
        """
        depth += 1
        data.update(data_atom)
        if depth == self.depth:
            return
        data["dict_"] = {}
        self.__fill_dict(depth, data_atom, data["dict_"])

    def __str__(self) -> str:
        return pformat(self.data)
