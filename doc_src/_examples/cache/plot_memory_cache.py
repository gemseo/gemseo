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
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.caches.memory_full_cache import MemoryFullCache
from numpy import array

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
#
# .. warning::
#     The :class:`.MemoryFullCache` relies on some multiprocessing features.
#     When working on Windows, the execution of scripts containing instances of
#     :class:`.MemoryFullCache` must be protected by an
#     ``if __name__ == '__main__':`` statement.

cache = MemoryFullCache()
print(cache)

###############################################################################
# Cache
# -----
#
# We can manually add data into the cache. However, it has to be noted that
# most of the time a cache is attached to an :class:`.MDODiscipline`. Then, the
# cache feeding has not to be performed explicitly by the user.

cache[{"x": array([1.0])}] = ({"y": array([2.0])}, None)
cache[{"x": array([2.0])}] = ({"y": array([3.0])}, None)
print(cache)

###############################################################################
# Get all data
# ------------
#
# Once the cache has been filled, the user can get the length of the cache. The
# user can also print all the data contained inside the cache.
print(len(cache))
for data in cache:
    print(data)

###############################################################################
# Get last cached data
# --------------------
#
# The user can access the last entry (inputs or outputs) which have been
# entered in the cache.

last_entry = cache.last_entry
print(last_entry.inputs)
print(last_entry.outputs)

###############################################################################
# Clear
# -----
# The user can clear an cache of all its entries by using the
# :meth:`.MemoryFullCache.clear` method:
cache.clear()
print(cache)
