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
"""# Access and clear a discipline cache

## Problem

Your discipline has been executed multiple times,
and results have been stored in your cache.
You want to access stored data, and at the end, clear the cache.

## Solution

GEMSEO provides mechanisms to interact with discipline's caches,
allowing you to retrieve cached data and to clear the cache when it is no longer needed.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.analytic import AnalyticDiscipline

# %%
# ### 1. Create and execute a discipline
#
# Here, you consider an analytic discipline.
# For the purpose of this example, set a full cache and
# execute the discipline a few times.
discipline = AnalyticDiscipline({"y": "x1+x2"})
discipline.set_cache(discipline.CacheType.MEMORY_FULL)
discipline.execute()
discipline.execute({"x1": array([2]), "x2": array([3])})
discipline.cache

# %%
# ### 2. Get cached data
#
# You can either extract all the cached data as a dataset...
discipline.cache.to_dataset()

# %%
# ... or iterate over the cache to retrieve every
# [CacheEntry][gemseo.caches.cache_entry.CacheEntry].
for inputs, entry in discipline.cache.items():
    print(inputs, entry)

# %%
#
# If you only want the last data (inputs, outputs and Jacobian),
# you can use the
# [last_item][gemseo.caches.base.BaseCache.last_item] property:
last_item = discipline.cache.last_item
print(f"Last inputs: {last_item.inputs}")
print(f"Last outputs: {last_item.outputs}")
print(f"Last jacobian: {last_item.jacobian}")

# %%
# ### 3. Clear the cache
discipline.cache.clear()
discipline.cache

# %%
# ## Summary
#
# - A cache stores one or many named tuples,
# called [CacheEntry][gemseo.caches.cache_entry.CacheEntry];
# - Data can be retrieved from a cache as a dataset with the
# [to_dataset()][gemseo.caches.base.BaseCache.to_dataset] method;
# - The last cached data is gotten with the
# [last_item][gemseo.caches.base.BaseCache.last_item] property;
# - The cache can be cleared with the
# [clear()][gemseo.caches.base.BaseCache.clear] method;
