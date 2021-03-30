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
"""
Python2 and Python3 compatibility layer
"""

import sys
from shutil import rmtree
from tempfile import mkdtemp

from numpy import array
from six import string_types

PY2 = sys.version_info.major == 2
PY3 = not PY2
PY36 = sys.version_info.major == 3 and sys.version_info.minor >= 6

if PY3:
    from builtins import int as _long

    string_dtype = "bytes"
    from builtins import range as xrange
    from inspect import getfullargspec as _getfullargspec

    from fastjsonschema import compile as compile_schema
    from fastjsonschema.exceptions import JsonSchemaException

    def next(iterator):
        return iterator.__next__()


else:
    from builtins import long as _long
    from builtins import xrange

    string_dtype = "string"
    from builtins import next
    from inspect import getargspec as _getfullargspec

    from gemseo.third_party.fastjsonschema import compile as compile_schema
    from gemseo.third_party.fastjsonschema.exceptions import JsonSchemaException

if PY36:
    OrderedDict = dict
else:
    from collections import OrderedDict


def is_py2():
    """Check if the version of Python is 2.

    :return: True if Python 2.
    :rtype: bool
    """
    return PY2


def long(data):
    """
    Return a long from data, performs casting
    """
    return _long(data)


def string_array(data):
    """
    Creates a numpy array of strings from data
    the dtype is adjusted (bytes in py3, string in py2)
    """
    return array(data, dtype=string_dtype)


if PY2:

    class TemporaryDirectory(object):
        """Create and return a temporary directory.  This has the same
        behavior as mkdtemp but can be used as a context manager.  For
        example:
            with TemporaryDirectory() as tmpdir:
                ...
        Upon exiting the context, the directory and everything contained
        in it are removed.

        Backported from python3 to python2 by Francois Gallard,
        with big simplification that may lead to some issues...
        """

        def __init__(self, suffix="", prefix="tmp", dir=None):
            self.name = None
            self.name = mkdtemp(suffix, prefix, dir)

        def __repr__(self):
            return "<{} {!r}>".format(self.__class__.__name__, self.name)

        def __enter__(self):
            return self.name

        def __exit__(self, exc, value, tb):
            self.cleanup()

        def __del__(self):
            self.cleanup()

        def cleanup(self):
            """ Delete the entire directory tree. """
            rmtree(self.name)


else:
    from tempfile import TemporaryDirectory

if PY2:
    getargspec = _getfullargspec

    def to_unicode_list(iterable):
        """ Convert a list of strings to a list of unicode strings."""
        return [s.decode("utf-8") for s in iterable]


else:

    def getargspec(func):
        """ Get arguments specifications. """
        return _getfullargspec(func)[:4]

    def to_unicode_list(iterable):
        """ Convert a list of strings to a list of unicode strings."""
        return iterable
