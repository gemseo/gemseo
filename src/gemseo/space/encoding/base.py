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

"""The base class of the encodings of a design space."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.space.design import DesignSpace
    from gemseo.util.typing import NumberArray


class BaseSpaceEncoding(metaclass=ABCGoogleDocstringInheritanceMeta):
    """An encoding of a design space.

    An encoding derives an *encoded* space from an *original* one,
    replacing some of its variables by others,
    possibly with another dimension.

    [encode][gemseo.space.encoding.base.BaseSpaceEncoding.encode]
    maps a value of the original space to the encoded space.
    [decode][gemseo.space.encoding.base.BaseSpaceEncoding.decode]
    maps any value of the encoded space to a value of the original space;
    several encoded values can be decoded as the same original one.
    [create_constraints][gemseo.space.encoding.base.BaseSpaceEncoding.create_constraints]
    returns the constraints that the encoded variables must satisfy.

    The encoded space is built once,
    from the original space as it is at that moment,
    and kept:
    an encoding describes a space as it was when the encoding was built,
    so a space that changes afterwards calls for a new encoding.
    """  # noqa: E501

    _original_space: DesignSpace
    """The space this encoding is built for."""

    __encoded_space: DesignSpace | None
    """The encoded space, once built."""

    def __init__(self, space: DesignSpace) -> None:
        """
        Args:
            space: The original space.
        """  # noqa: D205, D212
        self._original_space = space
        self.__encoded_space = None

    @property
    def original_space(self) -> DesignSpace:
        """The space this encoding is built for."""
        return self._original_space

    @property
    def encoded_space(self) -> DesignSpace:
        """The space where the encoded variables are replaced by others."""
        if self.__encoded_space is None:
            self.__encoded_space = self._create_encoded_space()

        return self.__encoded_space

    @abstractmethod
    def _create_encoded_space(self) -> DesignSpace:
        """Build the encoded space from the original one.

        Returns:
            The encoded space.
        """

    @abstractmethod
    def encode(self, value: NumberArray) -> NumberArray:
        """Encode a value of the original space.

        Args:
            value: A value of the original space,
                with any leading shape followed by the dimension of this space.

        Returns:
            The value of the encoded space that stands for it.
        """

    @abstractmethod
    def decode(self, value: NumberArray) -> NumberArray:
        """Decode a value of the encoded space.

        Args:
            value: A value of the encoded space,
                with any leading shape followed by the dimension of this space.

        Returns:
            The value of the original space chosen for it.
        """

    def create_constraints(self) -> tuple[ArrayFunction, ...]:
        """Create the constraints that the encoding imposes on the encoded space.

        Returns:
            The constraints,
            functions of a value of the encoded space,
            empty by default.
        """
        return ()
