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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory for caches."""
from __future__ import annotations

from gemseo.core.cache import AbstractCache
from gemseo.core.factory import Factory


class CacheFactory:
    """A factory for :class:`.AbstractCache`."""

    def __init__(self) -> None:  # noqa:D107
        self.factory = Factory(AbstractCache, ("gemseo.caches",))

    def create(self, cache_name: str, **options) -> AbstractCache:
        """Create an :class:`.AbstractCache`.

        Args:
            cache_name: The name of the cache class.
            **options: The options of the cache

        Returns:
            A cache.
        """
        return self.factory.create(cache_name, **options)

    @property
    def caches(self) -> list[str]:
        """The names of the cache classes."""
        return self.factory.classes

    def is_available(self, cache_name: str) -> bool:
        """Check the availability of a cache.

        Args:
            cache_name: The name of the cache.

        Returns:
            Whether the cache is available.
        """
        return self.factory.is_available(cache_name)
