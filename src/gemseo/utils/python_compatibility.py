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
from __future__ import annotations

import operator
import sys

if sys.version_info < (3, 8):  # pragma: >=3.8 no cover
    from typing_extensions import Final  # noqa: F401
    from typing_extensions import Literal  # noqa: F401

    def accumulate(iterable, func=operator.add, initial=None):
        """Accumulate implementation in plain Python.

        Args:
            iterable: An iterable sequence.
            func: An operator to apply on each element of the sequence.
            initial: The inital value of the accumulator.

        Yields:
            The accumulated item.

        Example:
            >>> accumulate([1,2,3,4,5])
            1 3 6 10 15
            >>> accumulate([1,2,3,4,5], initial=100)
            100 101 103 106 110 115
            >>> accumulate([1,2,3,4,5], operator.mul)
            1 2 6 24 120
        """
        it = iter(iterable)
        total = initial
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                return
        yield total
        for element in it:
            total = func(total, element)
            yield total

    import importlib_metadata  # noqa: F401

    from singledispatchmethod import singledispatchmethod  # noqa: F401

else:  # pragma: <3.8 no cover
    from functools import singledispatchmethod  # noqa: F401
    from importlib import metadata as importlib_metadata  # noqa: F401
    from itertools import accumulate  # noqa: F401
    from typing import Final  # noqa: F401
    from typing import Literal  # noqa: F401
