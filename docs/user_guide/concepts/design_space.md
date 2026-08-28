---
reading_time: true
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

A [DesignSpace][gemseo.space.design.DesignSpace]
is a collection of variables,
that can be either scalar or vector,
defined by bounds.
It is typically used
to define the input space that is explored through an optimization problem
or a design of experiments.

Each variable is described by:

- a name,
- a size (default: 1),
- a type ([read more][concept-variable-types]), either `"float"` (continuous, default) or `"integer"` (discrete),
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

## Variable types { #concept-variable-types }

The type of a variable is set with the `type_` argument
of [add_variable()][gemseo.space.design.DesignSpace.add_variable],
either as a string
or as a member of the `DesignSpace.DesignVariableType` enumeration
(also available as `gemseo.enum.DesignVariableType`).
[get_type()][gemseo.space.design.DesignSpace.get_type]
and [variable_types][gemseo.space.design.DesignSpace.variable_types]
read it back as a string.

Two types are available:

- `"float"` for the continuous variables (default),
- `"integer"` for the integer variables.

### Continuous variables { #concept-continuous-variables }

A continuous variable can take any real value between its bounds.

Think of the fuel mass loaded into an aircraft:
1250.5 kg is as meaningful as 1250.6 kg,
and so is any value in between.

This is the default type.
Its bounds default to $-\infty$ and $+\infty$,
and any finite bound is allowed.

When a variable has no value,
[initialize_missing_current_values()][gemseo.space.design.DesignSpace.initialize_missing_current_values]
gives each of its components
the middle of its bounds when both are finite,
the finite bound when only one of them is,
and zero otherwise.

!!! note
    A complex value is kept as is,
    so that the perturbation of a complex-step differentiation survives;
    see [to_complex()][gemseo.space.design.DesignSpace.to_complex].

### Integer variables { #concept-integer-variables }

An integer variable can only take integer values.

Think of the number of seats of a cabin:
it can be 180 or 181, never 180.5,
as half a seat does not exist.

The finite bounds must be integers too,
e.g. a cabin bounded between 150 and 200 seats;
otherwise a `ValueError` is raised.
Infinite bounds remain allowed.

Setting a non-integer value also raises a `ValueError`,
mentioning the type of the variable.

[round_vect()][gemseo.space.design.DesignSpace.round_vect]
rounds the integer components of a vector,
while [has_integer_variables][gemseo.space.design.DesignSpace.has_integer_variables]
and [get_integer_mask()][gemseo.space.design.DesignSpace.get_integer_mask]
tell where the integer variables are.

!!! warning
    An algorithm that does not declare that it handles integer variables
    rejects a problem including such variables;
    use the `skip_int_check` setting to force the execution.

??? abstract "API"

    - [add_variable()][gemseo.space.design.DesignSpace.add_variable]
    - [get_type()][gemseo.space.design.DesignSpace.get_type]
    - [variable_types][gemseo.space.design.DesignSpace.variable_types]
    - [has_integer_variables][gemseo.space.design.DesignSpace.has_integer_variables]
    - [get_integer_mask()][gemseo.space.design.DesignSpace.get_integer_mask]
    - [round_vect()][gemseo.space.design.DesignSpace.round_vect]

!!! how-to
    - [How to cast parameters into different types][]

## Integer relaxation { #concept-integer-relaxation }

Some algorithms only support continuous variables.
In that case,
the design space can relax integer variables by treating them as floats.

## Normalization of the variables { #concept-normalization-of-the-variables }

Optimization algorithms often work better when variables share a comparable scale,
that is why the design space can normalize bounded float variables $x$
into $x_{\mathrm{normalized}}$ in $[0, 1]$:

- $x_{\mathrm{normalized}} = \frac{x-l_b(x)}{u_b(x)-l_b(x)}$,
- $x_{\mathrm{normalized}} = \frac{x}{u_b(x)-l_b(x)}$,

where $l_b(x)$ and $u_b(x)$ are the lower and upper bounds of the variable $x$.

!!! warning
    Integer variables are not normalized.

!!! how-to
    - [How to (un)normalize design parameters][]

## Saving and loading { #concept-design-space-saving-loading}

A design space can be persisted to a file and reloaded later,
which is useful for sharing a problem definition or reusing a previous initial point.
Two formats are supported: [CSV](https://fr.wikipedia.org/wiki/Comma-separated_values) for human-readable exchange, and [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) for binary storage.

!!! how-to
    - [How to import and export a design space from disk][]

## Going further { #concept-going-further }

!!! how-to
    - [How to project parameters into boundaries][]
    - [How to reduce a design space][]
