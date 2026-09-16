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
"""A factory of variables."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from gemseo.core.base_factory import BaseFactory
from gemseo.space.variable.base import DataType
from gemseo.space.variable.deterministic import BaseDeterministicVariable
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from gemseo.util.pydantic import BaseSettings


class DeterministicVariableFactory(BaseFactory[BaseDeterministicVariable]):
    """A factory of deterministic variables.

    A random variable is built from the settings
    of the probability distributions of its components,
    not from a data type, so it is out of the scope of this factory.
    """

    _class: ClassVar[type[BaseDeterministicVariable]] = BaseDeterministicVariable
    _package_names: ClassVar[tuple[str, ...]] = ("gemseo.space.variable",)

    __data_type_to_class_name: dict[DataType, str]
    """The map from a data type to the name of the class pinning it."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.__data_type_to_class_name = {}

    @property
    def _data_type_to_class_name(self) -> dict[DataType, str]:
        """The map from a data type to the name of the class pinning it.

        Raises:
            ValueError: If two variable classes pin the same data type.
        """
        if not self.__data_type_to_class_name:
            data_type_to_class_name = {}
            for class_name in self.class_names:
                data_type = self.get_class(class_name).type
                other_class_name = data_type_to_class_name.get(data_type)
                if other_class_name is not None:
                    msg = (
                        f"The variable classes {other_class_name} and {class_name} "
                        f"both pin the data type {data_type}."
                    )
                    raise ValueError(msg)

                data_type_to_class_name[data_type] = class_name

            self.__data_type_to_class_name = data_type_to_class_name

        return self.__data_type_to_class_name

    def create_from_settings(  # noqa: D102
        self,
        settings: BaseSettings,
        *args: Any,
        **kwargs: Any,
    ) -> BaseDeterministicVariable:
        raise NotImplementedError

    @property
    def data_types(self) -> tuple[DataType, ...]:
        """The data types pinned by the variable classes."""
        return tuple(self._data_type_to_class_name)

    def get_class_from_data_type(
        self, data_type: DataType | str | bytes
    ) -> type[BaseDeterministicVariable]:
        """Return the variable class pinning a data type.

        Args:
            data_type: The type of the data of the variable.

        Returns:
            The variable class pinning the data type.

        Raises:
            ValueError: If `data_type` is not a data type
                or if no variable class pins it.
        """
        if isinstance(data_type, bytes):
            # An HDF file stores the type of a variable as bytes.
            data_type = data_type.decode()

        try:
            data_type = DataType(data_type)
        except ValueError:
            class_name = None
        else:
            class_name = self._data_type_to_class_name.get(data_type)

        if class_name is None:
            msg = (
                f"There is no variable class of type {data_type!r}; "
                "the available types are: "
                f"{pretty_str(self._data_type_to_class_name.keys())}."
            )
            raise ValueError(msg)

        return self.get_class(class_name)

    def create(
        self,
        data_type: DataType | str | bytes,
        *args: Any,
        **kwargs: Any,
    ) -> BaseDeterministicVariable:
        """Create a variable of a given data type.

        Args:
            data_type: The type of the data of the variable.

        Returns:
            The variable.

        Raises:
            ValueError: If `data_type` is not a data type
                or if no variable class pins it.
        """
        cls = self.get_class_from_data_type(data_type)
        return super().create(cls.__name__, *args, **kwargs)

    def update(self) -> None:  # noqa: D102
        super().update()
        # The data types are resolved from the discovered classes,
        # so a rediscovery invalidates the map.
        self.__data_type_to_class_name = {}


deterministic_variable_factory: Final[DeterministicVariableFactory] = (
    DeterministicVariableFactory()
)
"""The factory for `BaseDeterministicVariable` objects."""
