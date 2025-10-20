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
"""Decorators to lock and protect critical code sections."""

from __future__ import annotations

import functools
from multiprocessing import parent_process
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable


def synchronized(wrapped: Callable[..., Any]) -> Callable[..., Any]:
    """A synchronization decorator to avoid concurrent access of critical sections.

    The wrapped function must be a method of an object
    with a :attr:`_lock` attribute

    Args:
        wrapped: The function to be protected.
    """

    @functools.wraps(wrapped)
    def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Only need synchronization if we are in a multiprocessing context.
        if parent_process() is None:
            return wrapped(self, *args, **kwargs)
        with self._lock:
            return wrapped(self, *args, **kwargs)

    return _wrapper
