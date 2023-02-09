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
#        :author: Matthias De Lozzo
#        :author: Antoine Dechaume
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Pretty string utils."""
from __future__ import annotations

from collections import abc
from collections import namedtuple
from contextlib import contextmanager
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import Iterable

# to store the raw ingredients of a string to be formatted later
MessageLine = namedtuple("MessageLine", "str_format level args kwargs")

DEFAULT_DELIMITER = ", "
"""A string to separate string fields."""

DEFAULT_KEY_VALUE_SEPARATOR = "="
"""A string to separate key and value in a key-value pair of a mapping."""


def __stringify(
    obj: Any,
    delimiter: str,
    key_value_separator: str,
    function: Callable[[Any], str],
    sort: bool,
) -> str:
    """Represent an object with a string.

    Args:
        delimiter: The string to separate string fields.
        key_value_separator: The string to separate key and value
            in a key-value pair of a mapping.
        function: A function to represent an object with a string,
            e.g. :func:`str` or :func:`repr`.
        sort: Whether to sort the elements when the object if a collection.

    Returns:
        A string representing the object.
    """
    if not isinstance(obj, abc.Iterable):
        return function(obj)

    if isinstance(obj, abc.Mapping):
        obj = [
            f"{str(key)}{key_value_separator}{function(val)}"
            for key, val in obj.items()
        ]
    else:
        obj = [function(val) for val in obj]

    if sort:
        obj = sorted(obj)

    return delimiter.join(obj)


def pretty_repr(
    obj: Any,
    delimiter: str = DEFAULT_DELIMITER,
    key_value_separator: str = DEFAULT_KEY_VALUE_SEPARATOR,
    sort: bool = True,
) -> str:
    """Return an unambiguous string representation of an object based on :func:`repr`.

    Args:
        obj: The object to represent.
        delimiter: The string to separate string fields.
        key_value_separator: The string to separate key and value
            in a key-value pair of a mapping.
        sort: Whether to sort the elements when the object if a collection.

    Returns:
         An unambiguous string representation of the object.
    """
    return __stringify(obj, delimiter, key_value_separator, repr, sort)


def pretty_str(
    obj: Any,
    delimiter: str = DEFAULT_DELIMITER,
    key_value_separator: str = DEFAULT_KEY_VALUE_SEPARATOR,
    sort: bool = True,
) -> str:
    """Return a readable string representation of an object based on :func:`str`.

    Args:
        obj: The object to represent.
        delimiter: The string to separate string fields.
        key_value_separator: The string to separate key and value
            in a key-value pair of a mapping.
        sort: Whether to sort the elements when the object if a collection.

    Returns:
         A readable string representation of the object.
    """
    return __stringify(obj, delimiter, key_value_separator, str, sort)


def repr_variable(name: str, index: int, size: int = 0) -> str:
    """Return the string representation of a variable.

    Args:
        name: The name of the variable.
        index: The component of the variable.
        size: The size of the variable if known.
            Use ``0`` if unknown.

    Returns:
        The string representation of the variable.
    """
    return name if size == 1 else f"{name}[{index}]"


class MultiLineString:
    """Multi-line string lazy evaluator.

    The creation of the string is postponed to when an instance is stringified through
    the __repr__ method. This is mainly used for logging complex strings or objects
    where the string evaluation cost may be avoided when the logging level dismisses a
    logging message.

    A __add__ method is defined to allow the "+" operator between two instances,
    that implements the concatenation of two MultiLineString.
    If the other instance is not MultiLineString, it is first converted to string
    using its __str__ method and then added as a new line in the result.
    """

    INDENTATION = " " * 3
    DEFAULT_LEVEL = 0

    def __init__(
        self,
        lines: Iterable[MessageLine] | None = None,
    ) -> None:
        if lines is None:
            self.__lines = []
        else:
            self.__lines = list(lines)

        self.__level = None
        self.reset()

    def add(
        self,
        str_format: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add a line.

        Args:
            str_format: The string to be process by the format() method.
            args: The args passed to the format() method.
            kwargs: The kwargs passed to the format() method.
        """
        self.__lines.append(MessageLine(str_format, self.__level, args, kwargs))

    @property
    def lines(self) -> list[MessageLine]:
        """The strings composing the lines."""
        return self.__lines

    def reset(self) -> None:
        """Reset the indentation."""
        self.__level = self.DEFAULT_LEVEL

    def indent(self) -> None:
        """Increase the indentation."""
        self.__level += 1

    def dedent(self) -> None:
        """Decrease the indentation."""
        if self.__level > 0:
            self.__level -= 1

    def replace(
        self,
        old: str,
        new: str,
    ) -> MultiLineString:
        """Return a new MultiLineString with all occurrences of old replaced by new.

        Args:
            old: The sub-string to be replaced.
            new: The sub-string to be replaced with.

        Returns:
            The MultiLineString copy with replaced occurrences.
        """
        repl_msg = []
        for line in self.__lines:
            new_str = line.str_format.replace(old, new)
            repl_msg.append(MessageLine(new_str, line.level, line.args, line.kwargs))
        return MultiLineString(repl_msg)

    def __repr__(self) -> str:
        lines = []
        for line in self.__lines:
            str_format = self.INDENTATION * line.level + line.str_format
            lines.append(str_format.format(*line.args, **line.kwargs))
        return "\n".join(lines)

    def __add__(self, other: Any) -> MultiLineString:
        if isinstance(other, MultiLineString):
            return MultiLineString(self.lines + other.lines)
        out = deepcopy(self)
        out.add(str(other))
        return out

    @classmethod
    @contextmanager
    def offset(cls) -> None:
        cls.DEFAULT_LEVEL += 1
        try:
            yield
        finally:
            cls.DEFAULT_LEVEL -= 1
