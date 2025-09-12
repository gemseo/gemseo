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
"""A :class:`.Dataset` to store optimization histories."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from numpy import arange

from gemseo.datasets.dataset import Dataset
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.datasets.dataset import ComponentType
    from gemseo.datasets.dataset import DataType
    from gemseo.datasets.dataset import StrColumnType


class OptimizationDataset(Dataset):
    """A :class:`.Dataset` to store optimization histories."""

    DESIGN_GROUP: Final[str] = "designs"
    """The group name for the design variables."""

    # TODO: API: Remove this group name.
    FUNCTION_GROUP: Final[str] = "functions"
    """The group name for the functions.

    This group name is deprecated in favour of
    :attr:`.EQUALITY_CONSTRAINT_GROUP`,
    :attr:`.INEQUALITY_CONSTRAINT_GROUP`,
    :attr:`.OBJECTIVE_GROUP` and
    :attr:`.OBSERVABLE_GROUP`.
    """

    OBJECTIVE_GROUP: Final[str] = "objectives"
    """The group name for the objectives."""

    # TODO: API: Remove this group name.
    CONSTRAINT_GROUP: Final[str] = "constraints"
    """The group name for the constraints.

    This group name is deprecated in favour of
    :attr:`.EQUALITY_CONSTRAINT_GROUP` and
    :attr:`.INEQUALITY_CONSTRAINT_GROUP`.
    """

    INEQUALITY_CONSTRAINT_GROUP: Final[str] = "inequality_constraints"
    """The group name for the inequality constraints."""

    EQUALITY_CONSTRAINT_GROUP: Final[str] = "equality_constraints"
    """The group name for the equality constraints."""

    OBSERVABLE_GROUP: Final[str] = "observables"
    """The group name for the observables."""

    @property
    def _constructor(self) -> type[OptimizationDataset]:
        return OptimizationDataset

    @property
    def n_iterations(self) -> int:
        """The number of iterations."""
        return len(self)

    @property
    def iterations(self) -> list[int]:
        """The iterations."""
        return self.index.to_list()

    @property
    def design_variable_names(self) -> list[str]:
        """The names of the design variables.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return self.get_variable_names(self.DESIGN_GROUP)

    @property
    def constraint_names(self) -> list[str]:
        """The names of the constraints.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return self.get_variable_names(self.CONSTRAINT_GROUP)

    @property
    def equality_constraint_names(self) -> list[str]:
        """The names of the equality constraints.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return self.get_variable_names(self.EQUALITY_CONSTRAINT_GROUP)

    @property
    def inequality_constraint_names(self) -> list[str]:
        """The names of the inequality constraints.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return self.get_variable_names(self.INEQUALITY_CONSTRAINT_GROUP)

    @property
    def objective_names(self) -> list[str]:
        """The names of the objectives.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return self.get_variable_names(self.OBJECTIVE_GROUP)

    @property
    def observable_names(self) -> list[str]:
        """The names of the observables.

        Warnings:
            The names are sorted with the Python function ``sorted``.
        """
        return self.get_variable_names(self.OBSERVABLE_GROUP)

    @property
    def design_dataset(self) -> OptimizationDataset:
        """The view of the design dataset."""
        return self.get_view(group_names=self.DESIGN_GROUP)

    @property
    def constraint_dataset(self) -> OptimizationDataset:
        """The view of the constraint dataset."""
        return self.get_view(group_names=self.CONSTRAINT_GROUP)

    @property
    def equality_constraint_dataset(self) -> OptimizationDataset:
        """The view of the equality constraint dataset."""
        return self.get_view(group_names=self.EQUALITY_CONSTRAINT_GROUP)

    @property
    def inequality_constraint_dataset(self) -> OptimizationDataset:
        """The view of the inequality constraint dataset."""
        return self.get_view(group_names=self.INEQUALITY_CONSTRAINT_GROUP)

    @property
    def objective_dataset(self) -> OptimizationDataset:
        """The view of the objective dataset."""
        return self.get_view(group_names=self.OBJECTIVE_GROUP)

    @property
    def observable_dataset(self) -> OptimizationDataset:
        """The view of the observable dataset."""
        return self.get_view(group_names=self.OBSERVABLE_GROUP)

    def add_constraint_variable(
        self,
        variable_name: str,
        data: DataType,
        components: ComponentType = (),
    ) -> None:
        """Add data related to a constraint.

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
            group_name=self.CONSTRAINT_GROUP,
            components=components,
        )

    def add_equality_constraint_variable(
        self,
        variable_name: str,
        data: DataType,
        components: ComponentType = (),
    ) -> None:
        """Add data related to an equality constraint.

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
            group_name=self.EQUALITY_CONSTRAINT_GROUP,
            components=components,
        )

    def add_inequality_constraint_variable(
        self,
        variable_name: str,
        data: DataType,
        components: ComponentType = (),
    ) -> None:
        """Add data related to an inequality constraint.

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
            group_name=self.INEQUALITY_CONSTRAINT_GROUP,
            components=components,
        )

    def add_design_variable(
        self,
        variable_name: str,
        data: DataType,
        components: ComponentType = (),
    ) -> None:
        """Add data related to a design variable.

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
            group_name=self.DESIGN_GROUP,
            components=components,
        )

    def add_objective_variable(
        self,
        variable_name: str,
        data: DataType,
        components: ComponentType = (),
    ) -> None:
        """Add data related to an objective.

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
            group_name=self.OBJECTIVE_GROUP,
            components=components,
        )

    def add_observable_variable(
        self,
        variable_name: str,
        data: DataType,
        components: ComponentType = (),
    ) -> None:
        """Add data related to an observable.

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
            group_name=self.OBSERVABLE_GROUP,
            components=components,
        )

    def add_constraint_group(
        self,
        data: DataType,
        variable_names: StrColumnType = "c",
        variable_names_to_n_components: dict[str, int] | None = None,
    ) -> None:
        """Add the data related to the constraint group.

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
            self.CONSTRAINT_GROUP,
            data,
            variable_names=variable_names,
            variable_names_to_n_components=variable_names_to_n_components,
        )

    def add_equality_constraint_group(
        self,
        data: DataType,
        variable_names: StrColumnType = "c",
        variable_names_to_n_components: Mapping[str, int] = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """Add the data related to the equality constraint group.

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
            self.EQUALITY_CONSTRAINT_GROUP,
            data,
            variable_names=variable_names,
            variable_names_to_n_components=variable_names_to_n_components,
        )

    def add_inequality_constraint_group(
        self,
        data: DataType,
        variable_names: StrColumnType = "c",
        variable_names_to_n_components: Mapping[str, int] = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """Add the data related to the inequality constraint group.

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
            self.INEQUALITY_CONSTRAINT_GROUP,
            data,
            variable_names=variable_names,
            variable_names_to_n_components=variable_names_to_n_components,
        )

    def add_design_group(
        self,
        data: DataType,
        variable_names: StrColumnType = "d",
        variable_names_to_n_components: dict[str, int] | None = None,
    ) -> None:
        """Add the data related to the design variable group.

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
            self.DESIGN_GROUP,
            data,
            variable_names=variable_names,
            variable_names_to_n_components=variable_names_to_n_components,
        )

    def add_objective_group(
        self,
        data: DataType,
        variable_names: StrColumnType = "f",
        variable_names_to_n_components: dict[str, int] | None = None,
    ) -> None:
        """Add the data related to the objective group.

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
            self.OBJECTIVE_GROUP,
            data,
            variable_names=variable_names,
            variable_names_to_n_components=variable_names_to_n_components,
        )

    def add_observable_group(
        self,
        data: DataType,
        variable_names: StrColumnType = "o",
        variable_names_to_n_components: dict[str, int] | None = None,
    ) -> None:
        """Add the data related to the observable group.

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
            self.OBSERVABLE_GROUP,
            data,
            variable_names=variable_names,
            variable_names_to_n_components=variable_names_to_n_components,
        )

    def _reindex(self) -> None:
        self.index = arange(1, len(self) + 1)
