---
description: "A design space is a collection of bounded variables, each characterized by a name, a size, a type, bounds, and a current value."
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

# The design space { #concept-design-space }

A [DesignSpace][gemseo.algos.design_space.DesignSpace]
is a collection of variables,
that can be either scalar or vector,
defined by bounds.
It is typically used
to define the input space that is explored through an optimization problem
or a design of experiment/

Each variable is described by:

- a name,
- a size (default: 1),
- a type, either `"float"` (continuous, default) or `"integer"` (discrete),
- a lower bound (default: $-\infty$),
- an upper bound (default: $\infty$),
- a current value (default: none).

As an example,
when dealing with an aerodynamic simulation,
you might consider a continuous variable "wing_span"
bounded between 10 and 15 meters and
an integer variable "number_of_ribs" between 5 and 20.

A design space has several properties that allow you
to retrieve the information listed above.
It also includes a table view
that allows you to see all the variables at a glance.

!!! tutorial
    - [Tutorial - The design space][tutorial-the-design-space]

## Integer relaxation { #concept-integer-relaxation }

Some algorithms only support continuous variables.
In that case,
the design space can relax integer variables by treating them as floats.

## Normalization of the variables { #concept-normalization-of-the-variables }

Optimization algorithms often work better when variables share a comparable scale,
that is why the design space can normalize bounded float variables $x$
into $x_{\mathrm{normalized}}$ in $[0, 1]$:

- $x_{\mathrm{normalized} = \frac{x-l_b(x)}{u_b(x)-l_b(x)}}$,
- $x_{\mathrm{normalized}} = \frac{x}{u_b(x)-l_b(x)}$,

where $l_b(x)$ and $u_b(x)$ are the lower and upper bounds of the variable $x$.

!!! warning
    Integer variables are not normalized.

!!! how-to
    - [How to (un)normalize design parameters][how-to-unnormalize-design-parameters]

## Saving and loading { #concept-design-space-saving-loading}

A design space can be persisted to a file and reloaded later,
which is useful for sharing a problem definition or reusing a previous initial point.
Two formats are supported: [CSV](https://fr.wikipedia.org/wiki/Comma-separated_values) for human-readable exchange, and [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) for binary storage.

!!! how-to
    - [How to import and export a design space from disk][how-to-import-and-export-a-design-space-from-disk]

## Going further { #concept-going-further }

!!! how-to
    - [How to cast parameters into different types][how-to-cast-parameters-into-different-types]
    - [How to project parameters into boundaries][how-to-project-parameters-into-boundaries]
    - [How to reduce a design space][how-to-reduce-a-design-space]
