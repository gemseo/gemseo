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
"""# Merge different caches

## Problem

You execute the same discipline multiple times, in different workflows.
You have multiple [BaseFullCache][gemseo.caches.base_full.BaseFullCache],
and you would like to merge them into one.

## Solution

GEMSEO allows you to merge different caches.

!!! note
    These methods only apply to caches inheriting from
    [BaseFullCache][gemseo.caches.base_full.BaseFullCache].

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array
from numpy.random import default_rng

from gemseo.disciplines.analytic import AnalyticDiscipline

# %%
# ### 1. Create two disciplines and fill their cache
#
# Here, we consider the same discipline twice,
# and we execute them with random numbers so their cache are not the same.
rng = default_rng(seed=42)
expression = {"y": "2*x"}
disciplines = [AnalyticDiscipline(expression), AnalyticDiscipline(expression)]
for discipline in disciplines:
    discipline.set_cache(discipline.CacheType.MEMORY_FULL)
    discipline.execute({"x": array([rng.integers(0, 10)])})
    discipline.cache

# %%
# The first cache is...
disciplines[0].cache.to_dataset()

# %%
# ... and the second is
disciplines[1].cache.to_dataset()

# %%
# ### 2. Create a new cache
#
# You can create a new cache using the two caches:
cache = disciplines[0].cache + disciplines[1].cache
cache.to_dataset()

# %%
# ### 3. Merge into an existing cache
#
# You can also merge the first cache into the second one:
disciplines[1].cache.update(disciplines[0].cache)
disciplines[1].cache.to_dataset()

# %%
# ## Summary
#
# You can merge caches by:
#
# - creating a new cache and use the "+" operator;
# - updating an existing cache with the
# [update()][gemseo.caches.base_full.BaseFullCache.update] method.
