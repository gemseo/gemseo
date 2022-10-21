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
Simple cache
============
This example shows the manipulation of :class:`.SimpleCache` instances. This
cache only stores the last inputs and outputs stored.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.caches.simple_cache import SimpleCache
from numpy import array

configure_logger()

###############################################################################
# Import
# ------
# In the following lines, we import the `array` and the :class:`.SimpleCache`
# classes.


###############################################################################
# Create
# ------
# We can create an instance of the :class:`.SimpleCache` class with the
# following line:
cache = SimpleCache()

# The cache information can be displayed easily:
print(cache)

###############################################################################
# Cache
# -----
# It is possible to manually add some data into the cache by using the
# following lines:
cache[{"x": array([1.0])}] = ({"y": array([2.0])}, None)

###############################################################################
# We can add another entry to the cache, and we can then see that its length is
# still one. Indeed, as previously mentioned, the :class:`.SimpleCache` only
# enable to store one evaluation.
cache[{"x": array([2.0])}] = ({"y": array([3.0])}, None)
print(cache)

###############################################################################
# Get all data
# ------------
# We can display the lenght and the data contained in the cache. As mentionned
# before, we can see that only the last inputs and outputs cached are
# available:
print(len(cache))
for data in cache:
    print(data)

###############################################################################
# Get last cached data
# --------------------
# We can also print the last cached input and output data. For this cache, the
# last cached inputs and ouputs are also the only ones cached.

last_entry = cache.last_entry
print(last_entry.inputs)
print(last_entry.outputs)

###############################################################################
# Clear
# -----
# It is also possible to clear the cache, by using the following lines:
cache.clear()
print(cache)
