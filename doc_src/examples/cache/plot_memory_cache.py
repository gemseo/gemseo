# -*- coding: utf-8 -*-
# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Memory full cache
=================
This example shows how to manipulate a :class:`.MemoryFullCache` object.
"""
from __future__ import division, unicode_literals

from numpy import array

from gemseo.api import configure_logger
from gemseo.caches.memory_full_cache import MemoryFullCache

configure_logger()

###############################################################################
# Import
# ------
#
# First, we import the `array` and the :class:`MemoryError` classes.


###############################################################################
# Create
# ------
#
# We can create an instance of the :class:`.MemoryFullCache` class. We can then
# print it, and we can see it is empty.

cache = MemoryFullCache()
print(cache)

###############################################################################
# Cache
# -----
#
# We can manually add data into the cache. However, it has to be noted that
# most of the time a cache is attached to an :class:`.MDODiscipline`. Then, the
# cache feeding has not to be performed explicitly by the user.

data = {"x": array([1.0]), "y": array([2.0])}
cache.cache_outputs(data, ["x"], data, ["y"])
data = {"x": array([2.0]), "y": array([3.0])}
cache.cache_outputs(data, ["x"], data, ["y"])
print(cache)

###############################################################################
# Get all data
# ------------
#
# Once the cache has been filled, the user can get the length of the cache. The
# user can also print all the data contained inside the cache.
print(cache.get_length())
print(cache.get_all_data())

###############################################################################
# Get last cached data
# --------------------
#
# The user can access the last entry (inputs or outputs) which have been
# entered in the cache.

print(cache.get_last_cached_inputs())
print(cache.get_last_cached_outputs())

###############################################################################
# Clear
# -----
# The user can clear an cache of all its entries by using the
# :meth:`.MemoryFullCache.clear` method:
cache.clear()
print(cache)
