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
"""A :class:`.Dataset` to store input and output values."""

from __future__ import annotations

from typing import Final

from gemseo.datasets.dataset import ComponentType
from gemseo.datasets.dataset import Dataset
from gemseo.datasets.dataset import DataType
from gemseo.datasets.dataset import StrColumnType


class IODataset(Dataset):
    """A :class:`.Dataset` to store input and output values."""

    INPUT_GROUP: Final[str] = "inputs"
    """The group name for the input variables."""

    OUTPUT_GROUP: Final[str] = "outputs"
    """The group name for the output variables."""

    @property
    def _constructor(self) -> type[IODataset]:
        return IODataset

    def add_input_variable(
        self,
        variable_name: str,
        data: DataType,
        components: ComponentType = (),
    ) -> None:
        """Add data related to an input variable.

        Args:
            variable_name: The name of the variable.
            data: The data,
                either an array shaped as ``(n_entries, n_features)``,
                an array shaped as ``(n_entries,)``
                that will be reshaped as ``(n_entries, 1)``
                or a scalar that will be converted into an array
                shaped as ``(n_entries, 1)``.
            components: The components considered.
               If empty, use ``[0, ..., n_features]``.
        """
        self.add_variable(
            variable_name,
            data,
            group_name=self.INPUT_GROUP,
            components=components,
        )

    def add_output_variable(
        self,
        variable_name: str,
        data: DataType,
        components: ComponentType = (),
    ) -> None:
        """Add data related to an output variable.

        Args:
            variable_name: The name of the variable.
            data: The data,
                either an array shaped as ``(n_entries, n_features)``,
                an array shaped as ``(n_entries,)``
                that will be reshaped as ``(n_entries, 1)``
                or a scalar that will be converted into an array
                shaped as ``(n_entries, 1)``.
            components: The components considered.
               If empty, use ``[0, ..., n_features]``.
        """
        self.add_variable(
            variable_name,
            data,
            group_name=self.OUTPUT_GROUP,
            components=components,
        )

    def add_input_group(
        self,
        data: DataType,
        variable_names: StrColumnType = (),
        variable_names_to_n_components: dict[str, int] | None = None,
    ) -> None:
        """Add the data related to the input group.

        Args:
            data: The data.
            variable_names: The names of the variables.
                If empty, use :attr:`.DEFAULT_VARIABLE_NAME`.
            variable_names_to_n_components: The number of components of the variables.
                If ``variable_names`` is empty,
                this argument is not considered.
                If ``None``,
                assume that all the variables have a single component.
        """
        self.add_group(
            self.INPUT_GROUP,
            data,
            variable_names=variable_names,
            variable_names_to_n_components=variable_names_to_n_components,
        )

    def add_output_group(
        self,
        data: DataType,
        variable_names: StrColumnType = (),
        variable_names_to_n_components: dict[str, int] | None = None,
    ) -> None:
        """Add the data related to the output group.

        Args:
            data: The data.
            variable_names: The names of the variables.
                If empty, use :attr:`.DEFAULT_VARIABLE_NAME`.
            variable_names_to_n_components: The number of components of the variables.
                If ``variable_names`` is empty,
                this argument is not considered.
                If ``None``,
                assume that all the variables have a single component.
        """
        self.add_group(
            self.OUTPUT_GROUP,
            data,
            variable_names=variable_names,
            variable_names_to_n_components=variable_names_to_n_components,
        )

    @property
    def input_names(self) -> list[str]:
        """The names of the inputs.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return self.get_variable_names(self.INPUT_GROUP)

    @property
    def output_names(self) -> list[str]:
        """The names of the outputs.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return self.get_variable_names(self.OUTPUT_GROUP)

    @property
    def input_dataset(self) -> IODataset:
        """The view of the input dataset."""
        return self.get_view(group_names=self.INPUT_GROUP)

    @property
    def output_dataset(self) -> IODataset:
        """The view of the output dataset."""
        return self.get_view(group_names=self.OUTPUT_GROUP)

    @property
    def n_samples(self) -> int:
        """The number of samples."""
        return len(self)

    @property
    def samples(self) -> list[int | str]:
        """The ordered samples."""
        return self.index.to_list()
