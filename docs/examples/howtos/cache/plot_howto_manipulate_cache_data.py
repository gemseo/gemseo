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
"""# Manipulate data in a cache

## Problem

You have a cache, and you want to modify data it contains.

## Solution

GEMSEO allows you to manipulate cache data.
You can either modify the values
in an existing [CacheEntry][gemseo.caches.cache_entry.CacheEntry],
or you can add new [CacheEntry][gemseo.caches.cache_entry.CacheEntry].

!!! warning
    These methods must be used carefully.
    Changing cache data may produce workflow errors,
    since you are allowed to insert inconsistent/unprocessed data.

    For instance, in the $y=2*x$ AnalyticDiscipline,
    you can add the inconsistent entry `{x: 1, y:5}`.
    In that case, the discipline will return `y=5` for `x=1` using the cache,
    even if `5 != 2*1`.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.analytic import AnalyticDiscipline

# %%
# ### 1. Create and execute a discipline
#
# Here, we consider an analytic discipline.
# For the purpose of this example, let's set a full cache and
# execute the discipline a few times.
discipline = AnalyticDiscipline({"y": "2*x"})
discipline.set_cache(discipline.CacheType.MEMORY_FULL)
discipline.execute()
discipline.execute({"x": array([2])})
discipline.cache

# %%
# ### 2. Change a cache entry
#
# You can change an existing entry.
# To change the result when $x=2$, you have to give the tuple (output, jacobian):
discipline.cache[{"x": array([2])}] = ({"y": array([8])}, None)
discipline.cache

# %%
# ### 3. Add a new cache entry
#
# Without executing the discipline, you can add new entries.
# This can be used, for instance, to merge different caches of the same discipline,
# or to add known values.
#
# !!! note
#     When merging different caches into one,
#     you can iterate over a cache to get all its entries.
#     Details on [Access and clear a discipline cache][access-and-clear-a-discipline-cache].
discipline.cache[{"x": array([5])}] = ({"y": array([10])}, None)
discipline.cache[{"x": array([10])}] = ({"y": array([20])}, None)
discipline.cache

# %%
# ## Summary
#
# Manipulating data in a cache is as simple as manipulating data in a dictionary.
# The same syntax can be used to modify the output of an existing entry
# or to add a new one.
#
# This method should be used with care, as it may lead to incorrect
# discipline behavior.
