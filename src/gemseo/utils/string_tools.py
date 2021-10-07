# -*- coding: utf-8 -*-
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
"""
Pretty string utils
===================
"""
from collections import namedtuple
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, List, Optional

from gemseo.utils.py23_compat import PY2

if PY2:
    from collections import Iterable, Mapping
else:
    from collections.abc import Iterable, Mapping

# to store the raw ingredients of a string to be formatted later
MessageLine = namedtuple("MessageLine", "str_format level args kwargs")


def pretty_repr(
    obj,  # type: Any
    **kwargs  # type: Any
):  # type: (...)-> str
    """String representation of an object.

    Args:
        obj: The object to represent.

    Returns:
         A pretty string representation.
    """
    delimiter = kwargs.get("delimiter", ", ")

    if isinstance(obj, Mapping):
        return delimiter.join(
            ["{}={}".format(key, repr(val)) for key, val in sorted(obj.items())]
        )

    if isinstance(obj, Iterable):
        return delimiter.join([str(val) for val in obj])

    return repr(obj)


class MultiLineString(object):
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
        lines=None,  # type: Optional[Iterable[MessageLine]]
    ):  # type: (...) -> None
        if lines is None:
            self.__lines = []
        else:
            self.__lines = list(lines)

        self.__level = None
        self.reset()

    def add(
        self,
        str_format,  # type: str
        *args,  # type: Any
        **kwargs  # type: Any
    ):  # type: (...)-> None
        """Add a line.

        Args:
            str_format: The string to be process by the format() method.
            args: The args passed to the format() method.
            kwargs: The kwargs passed to the format() method.
        """
        self.__lines.append(MessageLine(str_format, self.__level, args, kwargs))

    @property
    def lines(self):  # type: (...) -> List[MessageLine]
        """The strings composing the lines."""
        return self.__lines

    def reset(self):  # type: (...) ->  None
        """Reset the indentation."""
        self.__level = self.DEFAULT_LEVEL

    def indent(self):  # type: (...) ->  None
        """Increase the indentation."""
        self.__level += 1

    def dedent(self):  # type: (...) ->  None
        """Decrease the indentation."""
        if self.__level > 0:
            self.__level -= 1

    def replace(
        self,
        old,  # type: str
        new,  # type: str
    ):  # type: (...) -> MultiLineString
        """Return a new MultiLineString with all occurrences of old replaced by new.

        Args:
            old: The sub-string to be replaced.
            new: The sub-string to be replaced with.

        Returns:
            The MultiLineString copy with replaced occurences.
        """
        repl_msg = []
        for line in self.__lines:
            new_str = line.str_format.replace(old, new)
            repl_msg.append(MessageLine(new_str, line.level, line.args, line.kwargs))
        return MultiLineString(repl_msg)

    def __repr__(self):  # type: (...) -> str
        lines = []
        for line in self.__lines:
            str_format = self.INDENTATION * line.level + line.str_format
            lines.append(str_format.format(*line.args, **line.kwargs))
        return "\n".join(lines)

    def __add__(
        self, other  # type: Any
    ):  # type: (...) -> MultiLineString
        if isinstance(other, MultiLineString):
            return MultiLineString(self.lines + other.lines)
        out = deepcopy(self)
        out.add(str(other))
        return out

    @classmethod
    @contextmanager
    def offset(cls):  # type: (...) ->  None
        cls.DEFAULT_LEVEL += 1
        try:
            yield
        finally:
            cls.DEFAULT_LEVEL -= 1
