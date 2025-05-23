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
This example shows how to manipulate an :class:`.MemoryFullCache` object.
"""

from __future__ import annotations

from numpy import array

from gemseo import configure_logger
from gemseo.caches.memory_full_cache import MemoryFullCache

configure_logger()

# %%
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
cache

# %%
# Cache
# -----
#
# We can manually add data into the cache. However, it has to be noted that
# most of the time a cache is attached to an :class:`.Discipline`. Then, the
# cache feeding has not to be performed explicitly by the user.

cache[{"x": array([1.0])}] = ({"y": array([2.0])}, None)
cache[{"x": array([2.0])}] = ({"y": array([3.0])}, None)
cache

# %%
# Get all data
# ------------
#
# We can now print some information from the cache, such as its length:
len(cache)

# %%
# We can
# also display all the cached data so far.
list(cache)

# %%
# Get last cached data
# --------------------
#
# The user can access the last entry (inputs or outputs) which have been
# entered in the cache.

last_entry = cache.last_entry
last_entry.inputs, last_entry.outputs

# %%
# Clear
# -----
# The user can clear a cache of all its entries by using the
# :meth:`.MemoryFullCache.clear` method:
cache.clear()
cache
