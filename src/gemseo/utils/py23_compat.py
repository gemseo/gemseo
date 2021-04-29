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

import sys

from numpy import array
from six import PY2, PY3, string_types  # noqa: F401

PY36 = PY3 and sys.version_info.minor >= 6

if PY3:
    from builtins import int as _long

    string_dtype = "bytes"
    from builtins import range as xrange
    from inspect import getfullargspec as _getfullargspec
    from pathlib import Path

    from fastjsonschema import compile as compile_schema
    from fastjsonschema.exceptions import JsonSchemaException

else:
    from builtins import long as _long
    from builtins import xrange  # noqa: F401

    string_dtype = "string"
    from builtins import next  # noqa: F401
    from inspect import getargspec as _getfullargspec

    from pathlib2 import Path  # noqa: F401

    from gemseo.third_party.fastjsonschema import (  # noqa: F401
        compile as compile_schema,
    )
    from gemseo.third_party.fastjsonschema.exceptions import (  # noqa: F401
        JsonSchemaException,
    )

if PY36:
    OrderedDict = dict
else:
    from collections import OrderedDict  # noqa: F401


def long(data):
    """Return a long from data, performs casting."""
    return _long(data)


def string_array(data):
    """Creates a numpy array of strings from data the dtype is adjusted (bytes in py3,
    string in py2)"""
    return array(data, dtype=string_dtype)


if PY2:
    getargspec = _getfullargspec

    def to_unicode_list(iterable):
        """Convert a list of strings to a list of unicode strings."""
        return [s.decode("utf-8") for s in iterable]


else:

    def getargspec(func):
        """Get arguments specifications."""
        return _getfullargspec(func)[:4]

    def to_unicode_list(iterable):
        """Convert a list of strings to a list of unicode strings."""
        return iterable
