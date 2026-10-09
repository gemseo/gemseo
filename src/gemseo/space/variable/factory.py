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
from gemseo.space.variable.base import VariableType
from gemseo.space.variable.deterministic import BaseDeterministicVariable
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from gemseo.util.pydantic import BaseSettings


class DeterministicVariableFactory(BaseFactory[BaseDeterministicVariable]):
    """A factory of deterministic variables.

    A random variable is built from the settings
    of the probability distributions of its components,
    not from a variable type, so it is out of the scope of this factory.
    """

    _class: ClassVar[type[BaseDeterministicVariable]] = BaseDeterministicVariable
    _package_names: ClassVar[tuple[str, ...]] = ("gemseo.space.variable",)

    __variable_type_to_class_name: dict[VariableType, str]
    """The map from a variable type to the name of the class pinning it."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.__variable_type_to_class_name = {}

    @property
    def _variable_type_to_class_name(self) -> dict[VariableType, str]:
        """The map from a variable type to the name of the class pinning it.

        Raises:
            ValueError: If two variable classes pin the same variable type.
        """
        if not self.__variable_type_to_class_name:
            variable_type_to_class_name = {}
            for class_name in self.class_names:
                variable_type = self.get_class(class_name).type
                other_class_name = variable_type_to_class_name.get(variable_type)
                if other_class_name is not None:
                    msg = (
                        f"The variable classes {other_class_name} and {class_name} "
                        f"both pin the variable type {variable_type}."
                    )
                    raise ValueError(msg)

                variable_type_to_class_name[variable_type] = class_name

            self.__variable_type_to_class_name = variable_type_to_class_name

        return self.__variable_type_to_class_name

    def create_from_settings(  # noqa: D102
        self,
        settings: BaseSettings,
        *args: Any,
        **kwargs: Any,
    ) -> BaseDeterministicVariable:
        raise NotImplementedError

    @property
    def variable_types(self) -> tuple[VariableType, ...]:
        """The variable types pinned by the variable classes."""
        return tuple(self._variable_type_to_class_name)

    def get_class_from_variable_type(
        self, variable_type: VariableType | str | bytes
    ) -> type[BaseDeterministicVariable]:
        """Return the variable class pinning a variable type.

        Args:
            variable_type: The type of the variable.

        Returns:
            The variable class pinning the variable type.

        Raises:
            ValueError: If `variable_type` is not a variable type
                or if no variable class pins it.
        """
        if isinstance(variable_type, bytes):
            # An HDF file stores the type of a variable as bytes.
            variable_type = variable_type.decode()

        try:
            variable_type = VariableType(variable_type)
        except ValueError:
            class_name = None
        else:
            class_name = self._variable_type_to_class_name.get(variable_type)

        if class_name is None:
            msg = (
                f"There is no variable class of type {variable_type!r}; "
                "the available types are: "
                f"{pretty_str(self._variable_type_to_class_name.keys())}."
            )
            raise ValueError(msg)

        return self.get_class(class_name)

    def create(
        self,
        variable_type: VariableType | str | bytes,
        *args: Any,
        **kwargs: Any,
    ) -> BaseDeterministicVariable:
        """Create a variable of a given variable type.

        Args:
            variable_type: The type of the variable.

        Returns:
            The variable.

        Raises:
            ValueError: If `variable_type` is not a variable type
                or if no variable class pins it.
        """
        cls = self.get_class_from_variable_type(variable_type)
        return super().create(cls.__name__, *args, **kwargs)

    def update(self) -> None:  # noqa: D102
        super().update()
        # The variable types are resolved from the discovered classes,
        # so a rediscovery invalidates the map.
        self.__variable_type_to_class_name = {}


deterministic_variable_factory: Final[DeterministicVariableFactory] = (
    DeterministicVariableFactory()
)
"""The factory for `BaseDeterministicVariable` objects."""
