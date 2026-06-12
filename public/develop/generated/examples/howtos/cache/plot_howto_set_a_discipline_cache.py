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
"""# Set a discipline cache

## Problem

You want to set the cache of your discipline.

## Solution

The cache of a discipline can be set with the
[set_cache()][gemseo.core.discipline.base_discipline.BaseDiscipline.set_cache] method.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.disciplines.analytic import AnalyticDiscipline

# %%
# ### 1. Create a discipline
#
# Here, we consider an analytic discipline.
discipline = AnalyticDiscipline({"y": "x1+x2"})
discipline.cache
# %%
# ### 2. Enumerate possible caches
#
# By default, a cache of type `SimpleCache` is attached to the discipline
# but different types of cache can be set for this discipline.
# The enumeration of the types of cache is retrieved by
# [CacheType][gemseo.core.discipline.base_discipline.CacheType].
list(discipline.CacheType)

# %%
# !!! warning
#     Some caches have usage restrictions.
#     Do not hesitate to read the [user guide][concept-different-cache-types].
#
# ### 3. Set another cache
#
# From that list, you can reset the cache with the
# [set_cache()][gemseo.core.discipline.base_discipline.BaseDiscipline.set_cache] method.
discipline.set_cache(discipline.CacheType.HDF5)
discipline.cache

# %%
# ## Summary
#
# The cache of a discipline can be reset with the
# [set_cache()][gemseo.core.discipline.base_discipline.BaseDiscipline.set_cache] method.
# The available cache policies are enumerated in
# [CacheType][gemseo.core.discipline.base_discipline.BaseDiscipline.CacheType].
