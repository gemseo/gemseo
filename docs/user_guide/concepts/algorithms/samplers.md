---
reading_time: true
complexity: beginner
status: draft
description: ""
tags: ['user_guide']
search:
  boost: 2
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Samplers (DOE) { #concept-samplers-doe }

## Algorithms { #concept-algorithms }

!!! note "Categorical variables"
    A DOE algorithm samples the unit hypercube
    and maps it onto the design space.
    In that mapping,
    the $n$ categories of a categorical variable
    split $[0, 1]$ into $n$ cells of length $1/n$:
    a coordinate $u$ in $[0, 1]$ gives
    the category at position $\lfloor n u \rfloor$
    (the last one when $u = 1$),
    and the category at position $i$
    is mapped to the centre $(i + 1/2)/n$ of its cell.
    So, each category has the same chance of being sampled.

## Advanced use { #concept-advanced-use }
