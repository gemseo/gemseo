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
"""A mutable sequence of functions."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import MutableSequence
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Functions(MutableSequence[MDOFunction]):
    """A mutable sequence of functions."""

    _functions: list[MDOFunction]
    """The functions."""

    _F_TYPES: ClassVar[tuple[MDOFunction.FunctionType]] = ()
    """The authorized types of functions.

    If empty, authorized all types.
    """

    evaluate_jacobian: bool
    """Whether the :meth:`.evaluate` method has to evaluate the Jacobian."""

    def __init__(self) -> None:  # noqa: D107
        self._functions = []
        self.evaluate_jacobian = False

    def __delitem__(self, key: int) -> None:
        del self._functions[key]

    def __getitem__(self, item: int) -> MDOFunction:
        return self._functions[item]

    def __setitem__(self, key: int, function: MDOFunction) -> None:
        self.__check_function_type(function)
        self._functions[key] = function

    def __len__(self) -> int:
        return len(self._functions)

    def insert(  # noqa: D102
        self, index: int | slice, function: MDOFunction | Iterable[MDOFunction]
    ) -> None:
        self.__check_function_type(function)
        self._functions.insert(index, function)

    def __check_function_type(
        self, function: MDOFunction | Iterable[MDOFunction]
    ) -> None:
        """Check if the function type is authorized.

        Args:
            function: The function(s).

        Raises:
            ValueError: When the function type is not authorized.
        """
        functions = [function] if isinstance(function, MDOFunction) else function
        for _function in functions:
            if self._F_TYPES and _function.f_type not in self._F_TYPES:
                msg = (
                    f"The function type '{_function.f_type}' is not "
                    f"one of those authorized ({pretty_str(self._F_TYPES)})."
                )
                raise ValueError(msg)

    def reset(self) -> None:
        """Reset the functions."""
        self._functions = list(self.get_originals())

    def format(self, function: MDOFunction) -> MDOFunction:
        """Format a function.

        Args:
            function: The function.

        Returns:
            A formatted function or ``None``.
        """
        return function

    def get_originals(self) -> Iterator[MDOFunction]:
        """Return the original functions.

        Yields:
            The original functions.
        """
        for function in self._functions:
            yield function.original

    def get_names(self) -> list[str]:
        """Return the names of the functions.

        Returns:
            The names of the functions.
        """
        return [function.name for function in self._functions]

    @property
    def dimension(self) -> int:
        """The sum of the output dimensions of the functions."""
        return self.get_dimension(self)

    @staticmethod
    def get_dimension(functions: Iterable[MDOFunction]) -> int:
        """Compute the sum of the output dimensions of functions.

        Args:
            functions: The functions.

        Returns:
            The dimension.

        Raises:
            ValueError: When the dimension of a function output is not available yet.
        """
        dimension = 0
        for function in functions:
            if not function.dim:
                msg = (
                    "The function output dimension is not available yet, "
                    f"please call function {function} once."
                )
                raise ValueError(msg)

            dimension += function.dim

        return dimension

    def evaluate(self, input_value: RealArray) -> None:
        """Evaluate all the functions given an input value.

        Args:
            input_value: The input value at which to evaluate the functions.

        .. note:: This method does not return the output values.
        """
        for function in self._functions:
            function.evaluate(input_value)
            if self.evaluate_jacobian:
                function.jac(input_value)

    @property
    def original_to_current_names(self) -> dict[str, list[str]]:
        """The current function names bound to the original ones."""
        original_to_current = {}
        for function in self._functions:
            if function.original_name not in original_to_current:
                original_to_current[function.original_name] = []
            original_to_current[function.original_name].append(function.name)

        return original_to_current

    def get_from_name(self, name: str, get_original: bool = False) -> MDOFunction:
        """Return a function from its name.

        Args:
            name: The name of the function.
            get_original: Whether to consider the original function.

        Returns:
            The function.

        Raises:
            ValueError: When there is no function with this name.
        """
        names = self.get_names()
        if name not in names:
            msg = (
                f"{name} is not among "
                f"the names of the {self.__class__.__name__.lower()}: "
                f"{pretty_str(names)}."
            )
            raise ValueError(msg)

        functions = self.get_originals() if get_original else self._functions
        return next(function for function in functions if function.name == name)
