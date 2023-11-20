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
Convert a cache to a dataset
============================

In this example,
we will see how to convert a cache to a :class:`.Dataset`.
"""

from __future__ import annotations

from numpy import array

from gemseo import configure_logger
from gemseo.caches.memory_full_cache import MemoryFullCache

configure_logger()

# %%
# Let us consider an :class:`.MemoryFullCache` storing two parameters:
#
# - x with dimension 1 which is a cache input,
# - y with dimension 2 which is a cache output.

cache = MemoryFullCache()
cache[{"x": array([1.0])}] = ({"y": array([2.0, 3.0])}, None)
cache[{"x": array([4.0])}] = ({"y": array([5.0, 6.0])}, None)

# %%
# This cache can be converted to an :class:`.IODataset`
# using its method :meth:`~.MemoryFullCache.to_dataset`:
dataset = cache.to_dataset("toy_cache")
dataset

# %%
# The input variables belong to the input group
# and the output variables to the output group.
# We can avoid this categorization and simply build a :class:`.Dataset`:
dataset = cache.to_dataset("toy_cache", categorize=False)
dataset
