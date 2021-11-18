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
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Python2 and Python3 compatibility layer."""

import operator
import sys
from typing import Callable, Iterable, List, Tuple

from numpy import array, ndarray
from six import PY2, PY3, string_types  # noqa: F401

if PY2:
    string_dtype = "string"
    xrange = xrange  # noqa: F821
    long = long  # noqa: F821

    from inspect import getargspec as _getfullargspec
    from re import match  # noqa: F401

    from backports.functools_lru_cache import lru_cache  # noqa: F401
    from pathlib2 import Path  # noqa: F401

    from gemseo.third_party.fastjsonschema import (  # noqa: F401
        compile as compile_schema,
    )
    from gemseo.third_party.fastjsonschema.exceptions import (  # noqa: F401
        JsonSchemaException,
    )

    getargspec = _getfullargspec

    def strings_to_unicode_list(
        iterable,  # type: Iterable[str]
    ):  # type: (...) -> List[str]
        """Convert a list of strings to a list of unicode strings."""
        return [s.decode("utf-8") for s in iterable]

    def fullmatch(pattern, string, flags=0):
        """Emulate python-3.4 re.fullmatch()."""
        return match(r"(?:" + pattern + r")\Z", string, flags=flags)

    import backports.unittest_mock

    backports.unittest_mock.install()


else:
    string_dtype = "bytes"
    long = int
    xrange = range

    from functools import lru_cache  # noqa: F401
    from inspect import getfullargspec as _getfullargspec
    from pathlib import Path  # noqa: F401
    from re import fullmatch  # noqa: F401
    from unittest import mock  # noqa: F401

    from fastjsonschema import compile as compile_schema  # noqa: F401
    from fastjsonschema.exceptions import JsonSchemaException  # noqa: F401

    def getargspec(
        func,  # type: Callable
    ):  # type: (...) -> Tuple[str]
        """Get arguments specifications."""
        return _getfullargspec(func)[:4]

    def strings_to_unicode_list(
        iterable,  # type: Iterable[str]
    ):  # type: (...) -> Iterable[str]
        """Convert a list of strings to a list of unicode strings."""
        return iterable


if sys.version_info < (3, 6):
    from collections import OrderedDict  # noqa: F401
else:
    OrderedDict = dict


def string_array(
    data,  # type: ndarray
):  # type: (...) -> ndarray
    """Creates a numpy array of strings from data the dtype is adjusted (bytes in py3,
    string in py2)"""
    return array(data, dtype=string_dtype)


if sys.version_info < (3, 8):

    def accumulate(iterable, func=operator.add, initial=None):
        """Accumulate implementation in plain Python.

        Args:
            iterable: An iterable sequence.
            func: An operator to apply on each element of the sequence.
            initial: The inital value of the accumulator.

        Yields:
            The accumulated item.

        Example:
            >>> accumulate([1,2,3,4,5])
            1 3 6 10 15
            >>> accumulate([1,2,3,4,5], initial=100)
            100 101 103 106 110 115
            >>> accumulate([1,2,3,4,5], operator.mul)
            1 2 6 24 120
        """
        it = iter(iterable)
        total = initial
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                return
        yield total
        for element in it:
            total = func(total, element)
            yield total

    if PY3:
        import importlib_metadata
    else:
        importlib_metadata = None

else:
    from importlib import metadata as importlib_metadata  # noqa: F401
    from itertools import accumulate  # noqa: F401
