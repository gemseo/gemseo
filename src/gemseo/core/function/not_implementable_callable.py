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
"""A non implemented callable object."""

from __future__ import annotations

from typing import Any
from typing import NoReturn


class NotImplementedCallable:
    """A callable object which raises NotImplementedError when called."""

    __message: str
    """The error message associated to the NotImplementedError."""

    def __init__(self, name: str, quantity_name: str) -> None:
        """
        Args:
            name: The name of the object to which the callable is attached.
            quantity_name: The name of the quantity of interest.
        """  # noqa: D205, D212
        self.__message = (
            f"The function computing the {quantity_name} of {name} is not implemented."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:  # noqa:D102
        raise NotImplementedError(self.__message)
