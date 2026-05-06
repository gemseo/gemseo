---
description: "Overview of mathematical problems in GEMSEO, from function evaluation to optimization, linear systems, and ODEs."
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

# Problems { #concept-problems }

Generally,
a problem can be defined as a question that needs an answer.
There may be zero, one, or more answers,
as well as zero, one or more ways to find them (or not!).

In GEMSEO,
a *problem* is a mathematical problem that can be solved using an *algorithm*,
e.g. an optimization problem solved by an optimizer.
The associated base class is [BaseProblem][gemseo.algos.base_problem.BaseProblem].

The following pages will show the different types of problems available in GEMSEO.
