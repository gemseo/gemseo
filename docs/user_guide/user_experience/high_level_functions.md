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

# High-level functions { #high-level-functions-user-guide }

The [gemseo][gemseo] module provides high-level functions for common workflows
without requiring in-depth knowledge of the code.

The API of these functions changes very rarely.
This is essential to ensure backward compatibility,
which means that scripts using it will work with future versions of GEMSEO without any changes.
This can be particularly useful if upgrading to a new major version involves significant costs.
However,
this should not be taken as a reason not to use the rest of the API.
The GEMSEO team is committed to supporting its users during API changes,
whether through a [dedicated page][upgrading-gemseo],
the [bump-gemseo](https://gitlab.com/gemseo/dev/bump-gemseo) tool to automatically apply most of the changes to your scripts,
or via the [discussion forum](https://gemseo.discourse.group/).

!!! note

    GEMSEO follows [semantic versioning](https://gemseo.gitlab.io/dev/gemseo-org/develop/contribute/developer_guide/#versioning) for its release numbering.
    An API change occurs when the first digit X of the version number X.Y.Z changes.

The main high-levels functions are:

- [create_discipline()][gemseo.create_discipline] to create a discipline,
- [create_design_space()][gemseo.create_design_space] to create an empty design space,
- [create_scenario()][gemseo.create_scenario] to create an MDO scenario from a design space and disciplines,
- [sample_disciplines()][gemseo.sample_disciplines] to sample disciplines from a design space,
- [execute_post()][gemseo.execute_post] to graphically analyze an evaluation history.

There are also specific functions for visualization:

- [generate_coupling_graph()][gemseo.generate_coupling_graph] to generate a coupling graph from disciplines,
- [generate_n2_plot()][gemseo.generate_n2_plot] to generate an N2 chart from disciplines,
- [generate_xdsm()][gemseo.generate_xdsm] to generate an XDSM from a scenario or a discipline.

We could also mention:

- [read_design_space()][gemseo.read_design_space] to read a design space,
- [write_design_space()][gemseo.write_design_space] to write a design space,
- [create_surrogate()][gemseo.create_surrogate] to create a surrogate discipline,
- [compute_doe()][gemseo.compute_doe] to create a design of experiments.

Please visit the [gemseo][gemseo] page for a complete overview of the high-level functions.
