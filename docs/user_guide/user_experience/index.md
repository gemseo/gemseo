---
description: ""
tags: []
search:
  boost: 1
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# User experience

The GEMSEO's [application programming interface (API)](https://en.wikipedia.org/wiki/API) serves as the interface
through which users harness the library's multidisciplinary capabilities.

This section highlights some key features of this API to maximize the user experience.

In Python,
an API comprises the classes, functions and variables
that developers invoke to build their applications.
GEMSEO's API is designed to balance power and usability,
offering both low-level control for developers and high-level convenience for end-users.

At its foundation,
GEMSEO embraces object-oriented design,
structuring the library around cohesive classes that model concepts,
most often in engineering and mathematics, and sometimes in software.
Thus,
disciplines become computational objects with inputs and outputs,
algorithms become configurable solvers with consistent interfaces,
design spaces become containers that define variable bounds and types,
and so on.
These concepts will be detailed in a [dedicated section][concepts-user-guide].

The object-oriented architecture brings several advantages:
components can be composed and reused,
behavior can be extended through inheritance,
and the separation of concerns keeps the codebase maintainable as complexity grows.

In the pages of this section,
we will present API features that apply across these concepts.
