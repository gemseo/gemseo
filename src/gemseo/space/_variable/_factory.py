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
from typing import Final

from gemseo.core.base_factory import BaseFactory
from gemseo.space._variable._base import BaseVariable
from gemseo.space._variable._base import DataType
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from gemseo.util.pydantic import BaseSettings


class VariableFactory(BaseFactory[BaseVariable]):
    """A factory of variables."""

    _CLASS = BaseVariable
    _PACKAGE_NAMES = ("gemseo.space._variable",)

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
                data_type = self.get_class(class_name).model_fields["type"].default
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

    def create_from_settings(
        self,
        settings: BaseSettings,
        *args: Any,
        **kwargs: Any,
    ) -> BaseVariable:
        raise NotImplementedError

    def create(
        self,
        data_type: DataType | str | bytes,
        *args: Any,
        **kwargs: Any,
    ) -> BaseVariable:
        """Create a variable of a given data type.

        Args:
            data_type: The type of the data of the variable.

        Returns:
            The variable.

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

        return super().create(class_name, *args, **kwargs)

    def update(self) -> None:  # noqa: D102
        super().update()
        # The data types are resolved from the discovered classes,
        # so a rediscovery invalidates the map.
        self.__data_type_to_class_name = {}


VARIABLE_FACTORY: Final[VariableFactory] = VariableFactory()
"""The factory for `Variable` objects."""
