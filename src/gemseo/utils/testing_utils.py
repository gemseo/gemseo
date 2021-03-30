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
#        :author: Damien Guenot
#                 Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Testing tools."""

import functools
from os import name as os_name

import pytest

from gemseo import LOGGER

IS_NT = os_name == "nt"

__skip_mp = pytest.mark.skipif(
    IS_NT, reason="Multiprocessing not available under Windows"
)


def __skip_nt(test_function):
    """
    Decorator used to skip a test under windows

    :param test_function: the function that is being tested and decorated
    :returns: the result of the wrapped function, if called, else None
    """
    if IS_NT:
        LOGGER.info("Skipping test %s", test_function.__name__)

        @functools.wraps(test_function)
        def wrapper(*args, **kwargs):  # pylint: disable=unused-argument
            pass

    else:

        @functools.wraps(test_function)
        def wrapper(*args, **kwargs):
            return test_function(*args, **kwargs)

    return wrapper


def skip_under_windows(test_function):
    """
    Decorator used to skip a test under windows
    Works both for unittests runs and pytest runs

    :param test_function: the function that is being tested and decorated
    :returns: the result of the wrapped function, if called, else None
    """
    return __skip_mp(__skip_nt(test_function))
