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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""This module contains a factory to instantiate a :class:`.AbstractCache` from its
class name.

The class can be internal to |g| or located in an external module whose path is provided
to the constructor. It also provides a list of available cache types and allows you to
test if a cache type is available.
"""
from __future__ import division, unicode_literals

import logging

from gemseo.core.cache import AbstractCache
from gemseo.core.factory import Factory

LOGGER = logging.getLogger(__name__)


class CacheFactory(object):
    """This factory instantiates a :class:`.AbstractCache` from its class name.

    The class can be internal to |g| or located in an external module whose path is
    provided to the constructor.
    """

    def __init__(self):
        """Initializes the factory: scans the directories to search for subclasses of
        AbstractCache.

        Searches in "|g|" and gemseo.caches
        """
        self.factory = Factory(AbstractCache, ("gemseo.caches",))

    def create(self, cache_name, **options):
        """Create a cache.

        :param str cache_name: name of the cache (its classname)
        :param options: additional options specific
        :return: cache_name cache

        Examples
        --------
        >>> from gemseo.caches.cache_factory import CacheFactory
        >>> cache = CacheFactory().create('MemoryFullCache', name='my_cache')
         my_cache
        |_ Type: MemoryFullCache
        |_ Input names: None
        |_ Output names: None
        |_ Length: 0
        |_ Tolerance: 0.0
        """
        return self.factory.create(cache_name, **options)

    @property
    def caches(self):
        """Lists the available classes.

        :returns: the list of classes names.
        :rtype: list(str)

        Examples
        --------
        >>> from gemseo.caches.cache_factory import CacheFactory
        >>> CacheFactory().caches
        ['AbstractFullCache', 'HDF5Cache', 'MemoryFullCache', 'SimpleCache']
        """
        return self.factory.classes

    def is_available(self, cache_name):
        """Checks the availability of a cache.

        :param str cache_name:  cache_name of the cache.
        :returns: True if the cache is available.
        :rtype: bool

        Examples
        --------
        >>> from gemseo.caches.cache_factory import CacheFactory
        >>> CacheFactory().is_available('SimpleCache')
        True
        >>> CacheFactory().is_available('UnavailableCache')
        False
        """
        return self.factory.is_available(cache_name)
