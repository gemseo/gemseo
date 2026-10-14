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
"""# Speed up a process made of inexpensive disciplines

## Problem

Your disciplines are inexpensive.
You do not need the cache, the data validation and the bound checks
that GEMSEO performs by default.

## Solution

Call the method `enable_fast_mode()` of the
[global configuration][concept-global-configuration].

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import configuration
from gemseo.discipline import AnalyticDiscipline

# %%
# ### 1. Enable the fast mode
#
# Do it before creating the disciplines,
# as a discipline chooses its cache when it is created.
configuration.enable_fast_mode()

# %%
# ### 2. Check the configuration
configuration.enable_discipline_cache, configuration.validate_input_data

# %%
discipline = AnalyticDiscipline({"y": "2*x"})
has_no_cache = discipline.cache is None
has_no_cache

# %%
# ### 3. Disable the fast mode
#
# This resets the options changed by the fast mode to their default values.
configuration.disable_fast_mode()
configuration.enable_discipline_cache, configuration.validate_input_data

# %%
# ## Summary
#
# Call `configuration.enable_fast_mode()` before creating the disciplines.
# It disables the cache, the data validation, the bound checks,
# the statistics, the statuses, the working database and the parallel execution.
#
# ## One step further
#
# To keep one of these options,
# set it after calling `enable_fast_mode()`:
# an option set afterwards keeps the value set.
