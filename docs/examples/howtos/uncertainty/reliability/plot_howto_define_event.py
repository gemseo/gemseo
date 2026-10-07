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
"""# Define an event

## Problem

You want to define the event of interest for a reliability analysis,
e.g. "the output exceeds a threshold",
possibly combining several conditions on several outputs.

## Solution

[EventVariable][gemseo.uncertainty.reliability.event_variable.EventVariable]
wraps the name of a variable of interest.
Comparing it to a threshold,
with `<`, `<=`, `>` or `>=`,
produces an
[Event][gemseo.uncertainty.reliability.event.Event].
Events combine with `&` (AND), `|` (OR) and `~` (NOT),
so a rich event can be built from elementary events.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.uncertainty.reliability.event_variable import EventVariable

# %%
# ### Prerequisites
#
# This how-to needs no discipline:
# an event variable only needs a name,
# so it can be defined,
# combined
# and evaluated on plain data,
# with no execution involved.
#
# Create three event variables named `f`, `g` and `h`:
f, g, h = EventVariable.from_names("f", "g", "h")

# %%
# ### 1. A comparison
#
# Comparing an event variable to a threshold,
# with `<`, `<=`, `>` or `>=`,
# creates an elementary event.
# Here the event is that `h` exceeds 400:
event = h > 400
event

# %%
# The reflected form reads the same way,
# from the threshold to the variable:
reflected_event = 400 < h
reflected_event

# %%
# Both forms give the same event:
assert reflected_event == event

# %%
# ### 2. An interval
#
# [isin()][gemseo.uncertainty.reliability.event_variable.EventVariable.isin]
# builds the event that a variable lies in a closed interval,
# i.e. both bounds are included:
event = h.isin([300, 500])
event

# %%
# ### 3. Combinations
#
# Events combine with `&` (AND) and `|` (OR).
# `&` takes precedence over `|`,
# exactly like `and` over `or` in plain Python,
# so `(f < 3) & (g > 4) | (h > 400)`
# reads as `((f < 3) & (g > 4)) | (h > 400)`.
# Each comparison must be parenthesized,
# because Python applies `&` and `|` before `<` and `>`:
event = (f < 3) & (g > 4) | (h > 400)
event

# %%
# !!! warning
#     Without the parentheses,
#     Python reads `f < 3 & g > 4` as `f < (3 & g) > 4`,
#     and `3 & g` raises a `TypeError`.
#
# ### 4. Negation
#
# `~` negates an event.
# On a single elementary event,
# the negation is exact,
# e.g. `~(h > 400)` is `h <= 400`:
event = ~(h > 400)
event

# %%
# On a combination,
# [De Morgan's laws](https://en.wikipedia.org/wiki/De_Morgan%27s_laws) apply,
# so the negation of an AND of elementary events becomes an OR of their negations,
# and conversely:
event = ~((f < 3) & (g > 4))
event

# %%
# !!! note
#     Negating a union of $N$ intersections of $n_1, \ldots, n_N$ elementary events
#     turns every intersection into a union of its negated elementary events,
#     with De Morgan's laws,
#     producing an intersection of unions;
#     re-expanding this intersection of unions
#     back into a union of intersections,
#     i.e. a disjunctive normal form (DNF),
#     yields up to $n_1 \times \ldots \times n_N$ intersections,
#     i.e. $n^N$ when all the intersections have $n$ elementary events,
#     which can grow very large;
#     [Event.max_intersections][gemseo.uncertainty.reliability.event.Event.max_intersections]
#     bounds this growth
#     and raises a `ValueError` if it would be exceeded.

# %%
# ### 5. What not to write
#
# Python evaluates a chained comparison, `and`, `or` and `not`
# by calling `bool()` on an intermediate result,
# which would silently drop part of the event.
# `Event` refuses this,
# raising a `TypeError` instead of returning a wrong result:
try:
    event = 2 < h < 5
except TypeError as error:
    print(error)

# %%
# Write `(2 < h) & (h < 5)`,
# or `h.isin([2, 5])`,
# instead of `2 < h < 5`.
# Write `&`, `|` and `~`,
# instead of `and`, `or` and `not`.

# %%
# ### 6. Check an event on data
#
# An event is a vectorized indicator function of its variables of interest.
# [evaluate()][gemseo.uncertainty.reliability.event.Event.evaluate]
# computes it from a mapping of variable names to arrays:
event = h > 400
data = {"h": array([350.0, 400.0, 450.0])}
event.evaluate(data)

# %%
# The first value, 350, does not exceed 400, so the indicator is 0;
# the last one, 450, does, so the indicator is 1;
# the middle one, 400, is the threshold itself,
# and `>` is strict, so it does not satisfy the event either, hence 0.
#
# ## Summary
#
# - [EventVariable][gemseo.uncertainty.reliability.event_variable.EventVariable]
#   wraps the name of a variable of interest;
# - comparing it with `<`, `<=`, `>`, `>=`,
#   or calling
#   [isin()][gemseo.uncertainty.reliability.event_variable.EventVariable.isin],
#   creates an
#   [Event][gemseo.uncertainty.reliability.event.Event];
# - [ThresholdComparator][gemseo.uncertainty.reliability.threshold_comparator.ThresholdComparator]
#   lists the four comparisons an event can hold;
# - events combine with `&`, `|` and `~`,
#   with `&` binding tighter than `|`;
# - a chained comparison, or the Python keywords `and`, `or`, `not`,
#   raise a `TypeError`;
# - [evaluate()][gemseo.uncertainty.reliability.event.Event.evaluate]
#   turns an event into a 0/1 indicator from data.
#
# ## One step further
#
# See
# [Run a reliability analysis](plot_howto_run_reliability_analysis.md)
# to use an event in a FORM analysis.
