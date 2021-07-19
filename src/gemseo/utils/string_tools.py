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

from gemseo.utils.py23_compat import PY2

if PY2:
    from collections import Iterable, Mapping
else:
    from collections.abc import Iterable, Mapping

# to store the raw ingredients of a string to be formatted later
MessageLine = namedtuple("MessageLine", "str_format level args kwargs")


def pretty_repr(obj, **kwargs):
    """String representation of an object.

    :param obj: object.
    :return: pretty string representation.
    :rtype: str
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
    """

    INDENTATION = " " * 3
    DEFAULT_LEVEL = 0

    def __init__(self):
        self.__lines = []
        self.__level = None
        self.reset()

    def add(self, str_format, *args, **kwargs):
        """Add a line.

        :param str str_format: string to be process by the format() method.
        :param args: args passed to the format() method.
        :param kwargs: kwargs passed to the format() method.
        """
        self.__lines.append(MessageLine(str_format, self.__level, args, kwargs))

    def reset(self):
        """Reset the indentation."""
        self.__level = self.DEFAULT_LEVEL

    def indent(self):
        """Increase the indentation."""
        self.__level += 1

    def dedent(self):
        """Decrease the indentation."""
        if self.__level > 0:
            self.__level -= 1

    def __repr__(self):
        lines = []
        for line in self.__lines:
            str_format = self.INDENTATION * line.level + line.str_format
            lines.append(str_format.format(*line.args, **line.kwargs))
        return "\n".join(lines)

    @classmethod
    @contextmanager
    def offset(cls):
        cls.DEFAULT_LEVEL += 1
        try:
            yield
        finally:
            cls.DEFAULT_LEVEL -= 1
