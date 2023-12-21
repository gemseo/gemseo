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
"""Implementation of the multiton design pattern."""

from __future__ import annotations

from abc import ABCMeta
from typing import Any
from typing import ClassVar

from docstring_inheritance import GoogleDocstringInheritanceMeta


class BaseMultiton(GoogleDocstringInheritanceMeta):
    """A metaclass implementing the multiton design pattern.

    See `Multiton <https://en.wikipedia.org/wiki/Multiton_pattern>`__.

    As opposed to the ``functools.lru_cache``,
    the objects built from this metaclass can be pickled.

    This metaclass has a cache
    which is a mapping of classes to class instances.
    When instantiating a class,
    if an instance has already been created for this class,
    then the cached instance is used,
    otherwise a new instance is created and stored into the cache.
    """

    __keys_to_class_instances: ClassVar[dict[Any, Any]] = {}
    """The cache that keeps the class instances."""

    def __call__(self) -> Any:  # noqa: D107 D102
        obj = self.__keys_to_class_instances.get(self)
        if obj is not None:
            return obj
        return self.__keys_to_class_instances.setdefault(self, type.__call__(self))

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cache."""
        cls.__keys_to_class_instances.clear()


class BaseABCMultiton(ABCMeta, BaseMultiton):
    """A metaclass of abstract classes implementing the multiton design pattern."""
