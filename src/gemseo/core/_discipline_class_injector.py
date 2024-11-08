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
"""Discipline base class injector."""

from __future__ import annotations

from importlib import import_module
from os import environ
from typing import Any
from typing import Final

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta


def _get_class(fully_qualified_name: str) -> type:
    """Return a class from its fully qualified name.

    Args:
        fully_qualified_name: The fully qualified name of the class.

    Returns:
        The class.
    """
    package_name, class_name = fully_qualified_name.rsplit(".", maxsplit=1)
    try:
        return getattr(import_module(package_name), class_name)
    except BaseException:  # noqa: BLE001
        msg = (
            f"The class {class_name} cannot be imported from the package "
            f"{package_name}."
        )
        raise ImportError(msg) from None


class ClassInjector(ABCGoogleDocstringInheritanceMeta):
    """A metaclass globally modifying the class hierarchy of a discipline.

    This enables to swap the original discipline base class (Discipline) for a new
    one without having to introduce complex and apparently useless abstractions.

    The environment variable :env:`GEMSEO_BASE_DISCIPLINE_CLASS` shall be set to the
    fully qualified name of the new base class, i.e. `package.sub_package.Class`.
    """

    __ORIGINAL_BASE_DISCIPLINE_CLASS_NAME: Final[str] = "Discipline"
    """The name of the original discipline class that should be changed in the class
    hierarchy of a derived class."""

    __NEW_BASE_DISCIPLINE_CLASS_QUAL_NAME: Final[str] = environ.get(
        "GEMSEO_BASE_DISCIPLINE_CLASS", ""
    )
    """The fully qualified name of the new discipline class that should replace the
    original one, if empty then no replacement is done."""

    if __NEW_BASE_DISCIPLINE_CLASS_QUAL_NAME:

        def __new__(
            cls,
            class_name: str,
            class_bases: tuple[type, ...],
            class_dict: dict[str, Any],
        ) -> Any:
            # Leave the original and new classes untouched when they are being created.
            if class_name not in (
                cls.__ORIGINAL_BASE_DISCIPLINE_CLASS_NAME,
                cls.__NEW_BASE_DISCIPLINE_CLASS_QUAL_NAME.split(".")[-1],
            ):
                # Find if and where is the original class is in the base classes.
                for _index, _base in enumerate(class_bases):
                    if _base.__name__ == cls.__ORIGINAL_BASE_DISCIPLINE_CLASS_NAME:
                        original_discipline_index = _index
                        break
                else:
                    original_discipline_index = None

                if original_discipline_index is not None:
                    new_class = _get_class(cls.__NEW_BASE_DISCIPLINE_CLASS_QUAL_NAME)

                    # Swap the original and new class in the base classes.
                    class_bases_ = list(class_bases)
                    class_bases_.pop(original_discipline_index)
                    class_bases_.insert(original_discipline_index, new_class)
                    class_bases = tuple(class_bases_)

            return super().__new__(cls, class_name, class_bases, class_dict)
