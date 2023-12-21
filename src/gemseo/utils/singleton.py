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
"""Implementations of the singleton design pattern."""

from __future__ import annotations

from os.path import realpath
from typing import Any
from typing import ClassVar


class SingleInstancePerAttributeId(type):
    """A multiton that depends on the id of a passed object.

    Subclasses are only instantiated when the discipline instance passed as input of the
    constructor is different from already created instances.

    The test if the instances are equal is made with the id(obj1)==id(obj2) operator
    """

    instances: ClassVar[dict[tuple[id, str], Any]] = {}

    # Eclipse is not happy with "cls" as first
    # argument but this is an eclipse bug.
    # function.MDODisciplineAdapterGenerator should have self as first parameter"
    def __call__(cls, *args, **kwargs):  # noqa:D102
        # id = memory address of the object, which is unique
        if not args:
            raise ValueError(
                "SingleInstancePerAttribute subclasses "
                "need at least one attribute in the constructor."
            )
        inst_key = (id(cls), id(args[0]))
        inst = cls.instances.get(inst_key)
        if inst is None:
            inst = type.__call__(cls, *args, **kwargs)
            cls.instances[inst_key] = inst
        return inst


class SingleInstancePerFileAttribute(type):
    """A multiton that depends on the file passed.

    Subclasses are only instantiated when the discipline instance passed as input of the
    constructor is different from already created instances.

    The test if the instances are equal is made with the obj1 == obj2 operator
    """

    instances: ClassVar[dict[tuple[id, str], Any]] = {}

    # Eclipse is not happy with "cls" as first
    # argument but this is an eclipse bug.
    def __call__(cls, *args, **kwargs):  # noqa:D102
        if not args:
            raise ValueError(
                "SingleInstancePerAttribute subclasses need at"
                " least one attribute in the constructor."
            )
        fpath = args[0]

        if not isinstance(fpath, str):
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
