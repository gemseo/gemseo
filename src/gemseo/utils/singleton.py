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
"""Singletons implementation and variants."""
from __future__ import annotations

from os.path import realpath
from typing import Any

from six import with_metaclass


class SingleInstancePerAttributeId(type):
    """A Singleton-like design pattern so that subclasses are only instantiated when the
    discipline instance passed as input of the constructor is different from already
    created instances.

    The test if the instances are equal is made with the id(obj1)==id(obj2) operator
    """

    instances = {}

    # Eclipse is not happy with "cls" as first
    # argument but this is an eclipse bug.
    # function.MDOFunctionGenerator should have self as first parameter"
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


class _Multiton(type):
    """A metaclass for implementing the Multiton design pattern.

    See `Multiton <https://en.wikipedia.org/wiki/Multiton_pattern>`.

    As opposed to the functools.lru_cache,
    the objects built from this metaclass can be pickled.

    .. warning:

        Like the standard functools.lru_cache,
        the kwargs order is not preserved:
        it means that f(x=1, y=2) is treated as a
        distinct call from f(y=2, x=1) which will be cached separately.
    """

    _cache: Any = {}

    def __call__(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        key = (cls,) + args + tuple(kwargs.items())
        try:
            return cls._cache[key]
        except KeyError:
            inst = type.__call__(cls, *args, **kwargs)
            cls._cache[key] = inst
            return inst

    @classmethod
    def cache_clear(cls) -> None:
        """Clear the cache."""
        cls._cache = {}


# Provide a naturally derivable class.
Multiton = with_metaclass(_Multiton, object)


class SingleInstancePerFileAttribute(type):
    """A Singleton-like design pattern so that subclasses are only instantiated when the
    discipline instance passed as input of the constructor is different from already
    created instances.

    The test if the instances are equal is made with the obj1 == obj2 operator
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
