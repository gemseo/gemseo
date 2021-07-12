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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""Wrapper of the multiprocessing module in order to circumvent the issues encountered
on Windows with the multiprocessing."""
from __future__ import division, unicode_literals

import logging
import os
import threading

LOGGER = logging.getLogger(__name__)


# On Windows, some functions and classes are monkey-patched in order to keep it
# compatible with older Python 2.7 interpreters. Particularly, issues have been
# encoutered with the 2.7.9 interpreter under Windows.
if os.name == "nt":

    class MockValue(object):
        """A mock object for multiprocessing Value."""

        def __init__(self, value):
            """Constructor.

            :param value: initializing value
            :type value: `int` or `float`
            """
            self.value = value
            self.lock = threading.RLock()

        def get_lock(self):
            """Get the Lock."""
            return self.lock

    def new_value(new_type, initial_value):
        """Monkey patch for the Value function.

        :param new_type: type of the Value, currently 'i' or 'd' are supported
        :type new_type: `str`
        :param initial_value: initial value of the Value object
        :type initial_value: `int` or `float`
        """
        if new_type == "i":
            initial_value = int(initial_value)
        elif new_type == "d":
            initial_value = float(initial_value)
        else:
            raise ValueError("%s is not a supported type" % new_type)

        return MockValue(initial_value)

    class NewManager(object):
        """Monkey patch for the Manager class."""

        @staticmethod
        def dict():
            """Monkey patch the dict() Manager object."""
            return dict()

    RLock = threading.RLock
    Manager = NewManager
    Value = new_value
else:
    from multiprocessing import Manager, RLock, Value  # noqa: F401
