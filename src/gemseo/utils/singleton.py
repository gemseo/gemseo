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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Singletons implementation and variants
**************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import str
from os.path import realpath

from future import standard_library

from gemseo.utils.py23_compat import string_types

standard_library.install_aliases()


class SingleInstancePerAttributeId(type):
    """A Singleton-like design pattern so that subclasses
    are only instantiated when the discipline instance
    passed as input of the constructor is different from
    already created instances.
    The test if the instances are equal is made with
    the id(obj1)==id(obj2) operator
    """

    instances = {}

    # Eclipse is not happy with "cls" as first
    # argument but this is an eclipse bug.
    # "function.MDOFunctionGenerator' should have self as first parameter"
    def __call__(cls, *args, **kwargs):
        # id = memory address of the object, which is unique
        if not args:
            raise ValueError(
                "SingleInstancePerAttribute subclasses "
                + "need at"
                + " least one attribute in the constructor."
            )
        inst_key = (id(cls), id(args[0]))
        inst = cls.instances.get(inst_key)
        if inst is None:
            inst = type.__call__(cls, *args, **kwargs)
            cls.instances[inst_key] = inst
        return inst


class SingleInstancePerAttributeEq(type):
    """A Singleton-like design pattern so that subclasses
    are only instantiated when the discipline instance
    passed as input of the constructor is different from
    already created instances.
    The test if the instances are equal is made with the
    obj1 == obj2 operator
    """

    instances = {}

    # Eclipse is not happy with "cls" as first
    # argument but this is an eclipse bug.
    # "function.MDOFunctionGenerator' should have self as first parameter"
    def __call__(cls, *args, **kwargs):
        if not args:
            args = [None]
        inst_key = (id(cls),) + tuple(args)
        inst = cls.instances.get(inst_key)
        if inst is None:
            inst = type.__call__(cls, *args, **kwargs)
            cls.instances[inst_key] = inst
        return inst


class SingleInstancePerFileAttribute(type):
    """A Singleton-like design pattern so that subclasses
    are only instantiated when the discipline instance
    passed as input of the constructor is different from
    already created instances.
    The test if the instances are equal is made with the
    obj1 == obj2 operator
    """

    instances = {}

    # Eclipse is not happy with "cls" as first
    # argument but this is an eclipse bug.
    def __call__(cls, *args, **kwargs):
        if not args:
            raise ValueError(
                "SingleInstancePerAttribute subclasses need at"
                + " least one attribute in the constructor."
            )
        fpath = args[0]

        if not isinstance(fpath, string_types):
            raise TypeError(
                "Argument 0 is not a string but of type :" + str(type(fpath))
            )
        fpath = realpath(fpath)
        inst_key = (id(cls), fpath)
        inst = cls.instances.get(inst_key)
        if inst is None:
            inst = type.__call__(cls, *args, **kwargs)
            cls.instances[inst_key] = inst
        return inst
