---
reading_time: true
complexity: beginner
description: "A design space is a collection of variables, each characterized by a name, a size, a type, bounds or a set of values, and a current value."
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
defined by bounds
or by the set of values they can take.
It is typically used
to define the input space that is explored through an optimization problem
or a design of experiments.

Each variable is described by:

- a name,
- a size (default: 1),
- a type ([read more][concept-variable-types]), either `"real"` (default), `"integer"`, `"discrete"`, `"catalog"` or `"categorical"`,
- a lower bound (default: $-\infty$), for a numeric variable,
- an upper bound (default: $\infty$), for a numeric variable,
- a current value (default: none).

As an example,
when dealing with an aerodynamic simulation,
you might consider a real variable "wing_span"
bounded between 10 and 15 meters and
an integer variable "number_of_ribs" between 5 and 20
and a categorical variable "material"
taking the value `"aluminium"`, `"titanium"` or `"composite"`.

A design space has several properties that allow you
to retrieve the information listed above.
It also includes a table view
that allows you to see all the variables at a glance.

!!! tutorial
    - [Tutorial - The design space][tutorial-the-design-space]

## Variable types { #concept-variable-types }

Four types are available:

- `"real"` for the real variables (default),
- `"integer"` for the integer variables,
- `"discrete"` for the discrete numeric variables,
- `"catalog"` for the variables choosing an alternative in a catalog.
- `"categorical"` for the variables whose value is one of a set of unordered labels.

Each type has its own declaration method,
taking what a variable of that type is defined by:
[add_real_variable()][gemseo.space.design.DesignSpace.add_real_variable]
and
[add_integer_variable()][gemseo.space.design.DesignSpace.add_integer_variable]
take a size and bounds,
[add_discrete_variable()][gemseo.space.design.DesignSpace.add_discrete_variable]
takes the values that the variable can take,
[add_catalog_variable()][gemseo.space.design.DesignSpace.add_catalog_variable]
the catalog of the alternatives
and
[add_categorical_variable()][gemseo.space.design.DesignSpace.add_categorical_variable]
the labels that the variable can take.

### Real variables { #concept-real-variables }

A real variable can take any real value between its bounds.

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
while [has_variables_of_type()][gemseo.space.variables_view.VariablesView.has_variables_of_type],
called with `DesignVariableType.INTEGER`,
and [get_integer_mask()][gemseo.space.design.DesignSpace.get_integer_mask]
tell where the integer variables are.

!!! warning
    An algorithm that does not declare that it handles integer variables
    (`handle_integer_variables`) rejects a problem including such variables;
    set `relax_integer_variables` to `True` to relax them to float variables
    and run it anyway.

??? abstract "API"

    - [add_real_variable()][gemseo.space.design.DesignSpace.add_real_variable]
    - [add_integer_variable()][gemseo.space.design.DesignSpace.add_integer_variable]
    - [add_discrete_variable()][gemseo.space.design.DesignSpace.add_discrete_variable]
    - [variables][gemseo.space.design.DesignSpace.variables]
    - [has_variables_of_type()][gemseo.space.variables_view.VariablesView.has_variables_of_type]
    - [get_integer_mask()][gemseo.space.design.DesignSpace.get_integer_mask]
    - [round_vect()][gemseo.space.design.DesignSpace.round_vect]

!!! how-to
    - [How to cast parameters into different types][]

### Discrete variables { #concept-discrete-variables }

A discrete variable can only take a limited number of numeric values,
called its choices.

Think of the number of plies of a composite laminate
chosen from a qualified catalogue,
or of a thickness that a supplier only delivers in $0.4$ mm and $0.47$ mm:
a value in between is not manufacturable,
even though it lies between the smallest and the largest one.

The choices are passed to
[add_discrete_variable()][gemseo.space.design.DesignSpace.add_discrete_variable];
they must be numbers,
and cannot be changed afterwards.
A discrete variable is scalar:
declare a vector of discrete quantities as several discrete variables.

The lower (resp. upper) bound is the smallest (resp. largest) choice;
it cannot be set manually.

When a variable has no value,
[initialize_missing_current_values()][gemseo.space.design.DesignSpace.initialize_missing_current_values]
gives it its first choice.

[variables][gemseo.space.design.DesignSpace.variables]
reads the choices back,
as `design_space.variables["thickness"].choices`,
while [has_variables_of_type()][gemseo.space.variables_view.VariablesView.has_variables_of_type],
called with `DesignVariableType.DISCRETE`,
tells whether the design space holds a discrete variable.

!!! warning
    An algorithm that does not declare that it handles discrete variables
    rejects a problem including such variables;
    set `relax_discrete_variables` to `True` to relax them to float variables
    and run it anyway.

!!! note
    The components of a discrete variable are stored as floats,
    so a variable whose choices are integers,
    e.g. `[2, 4, 6, 8]`,
    has `2.0` as first choice.

??? abstract "API"

    - [add_discrete_variable()][gemseo.space.design.DesignSpace.add_discrete_variable]
    - [variables][gemseo.space.design.DesignSpace.variables]
    - [has_variables_of_type()][gemseo.space.variables_view.VariablesView.has_variables_of_type]
    - [DiscreteVariable][gemseo.space.variable.DiscreteVariable]

### Categorical variables { #concept-categorical-variables }

A categorical variable takes different labels, also called categories.
Unlike the choices of a discrete variable,
these labels are not numbers
and have no order.

Think of the material of a wing panel,
chosen among `"aluminium"`, `"titanium"` and `"composite"`:
the materials cannot be sorted,
and there is no material halfway between two of them,
unlike the integer `2` between `1` and `3`.

The categories are passed to
[add_categorical_variable()][gemseo.space.design.DesignSpace.add_categorical_variable];
they must be strings without duplication,
and cannot be changed afterwards.
A categorical variable is scalar
and has no bounds.

A discipline receives the label,
e.g. `"titanium"`,
while the current value, the design vector and the database
store the position of this label among the categories,
starting from zero,
e.g. `1`.
[set_current_value()][gemseo.space.design.DesignSpace.set_current_value],
when passed a mapping,
and [set_current_variable()][gemseo.space.design.DesignSpace.set_current_variable]
accept the label as well as its position,
and the table view of the design space displays the label.

When a variable has no value,
[initialize_missing_current_values()][gemseo.space.design.DesignSpace.initialize_missing_current_values]
gives it its first category.

[variables][gemseo.space.design.DesignSpace.variables]
reads the categories back,
as `design_space.variables["material"].categories`,
while [has_variables_of_type()][gemseo.space.variables_view.VariablesView.has_variables_of_type],
called with `DesignVariableType.CATEGORICAL`,
tells whether the design space holds a categorical variable.

!!! warning
    The optimization algorithms reject a problem including a categorical variable,
    and no option relaxes it,
    as its categories have no order.
    The DOE algorithms sample it
    ([read more][concept-samplers-doe]).

!!! note
    When a design space is saved to a file,
    the value of a categorical variable is stored as its label,
    and the variable has no bounds to store:
    an HDF file has neither lower-bound nor upper-bound dataset.
    A design space with a categorical variable can only be saved to an HDF file,
    not to a CSV file.

??? abstract "API"

    - [add_categorical_variable()][gemseo.space.design.DesignSpace.add_categorical_variable]
    - [variables][gemseo.space.design.DesignSpace.variables]
    - [has_variables_of_type()][gemseo.space.variables_view.VariablesView.has_variables_of_type]
    - [CategoricalVariable][gemseo.space.variable.CategoricalVariable]

### Catalog variables { #concept-catalog-variables }

A catalog variable chooses one alternative in a **catalog**,
and its value is the **position** of that alternative in the catalog,
from $0$ to the number of alternatives minus one.

Think of a material picked from a qualified list,
of a supplier, or of an off-the-shelf component:
the alternatives are not ordered,
and what distinguishes them are their properties.

A catalog is a table:
one row per alternative,
one column per property,
and one index label naming each alternative.
It is built from a `pandas` table or from a mapping
from a property name to a sequence of values.

For instance, a catalog of three materials,
whose positions are given for the sake of clarity
but are not part of the catalog:

| Position | Alternative | `density` | `cost` |
|---------:|-------------|----------:|-------:|
| 0        | aluminium   | 2.7       | 10.0   |
| 1        | steel       | 7.8       | 5.0    |
| 2        | titanium    | 4.5       | 50.0   |

A variable `material` built from this catalog
takes the values $0$, $1$ and $2$,
standing for aluminium, steel and titanium.
Its value can also be given by label,
e.g. `"steel"` for $1$,
both to [add_catalog_variable()][gemseo.space.design.DesignSpace.add_catalog_variable]
and to [set_current_variable()][gemseo.space.design.DesignSpace.set_current_variable];
a label naming no alternative, or several ones, is rejected,
and the position must then be given instead,
while a label given to a variable that is neither a catalog
nor a categorical variable
raises a `TypeError`.
[Catalog.get_position()][gemseo.space.catalog.catalog.Catalog.get_position]
returns the position of the alternative named by a label.
The lower (resp. upper) bound is $0$ (resp. $2$);
it cannot be set manually.
Since its value is a position,
the design space handles it as an integer:
[get_integer_mask()][gemseo.space.design.DesignSpace.get_integer_mask] marks it,
[round_vect()][gemseo.space.design.DesignSpace.round_vect] rounds it,
and so does [denormalize_vect()][gemseo.space.design.DesignSpace.denormalize_vect]
with `minus_lb=True`.
A catalog variable is scalar:
declare several of them rather than a vector.
When a variable has no value,
[initialize_missing_current_values()][gemseo.space.design.DesignSpace.initialize_missing_current_values]
gives it the position $0$, i.e. the first alternative.

[variables][gemseo.space.design.DesignSpace.variables]
reads the catalog back,
as `design_space.variables["material"].catalog`,
while [has_variables_of_type()][gemseo.space.variables_view.VariablesView.has_variables_of_type],
called with `DesignVariableType.CATALOG`,
tells whether the design space holds a catalog variable.
The tabular view shows such a variable as a bounded integer;
it adds no column.

The catalog is **formatted** when the `Catalog` is built,
which `add_catalog_variable()` does from the properties it is given,
while a `Catalog` it is given is already formatted;
the steps run in this order:

1. the labels and the properties are read from the input:
   a string, a byte string or a zero-dimensional array given as the labels
   is a single label, not a sequence of characters,
   a `pandas` table whose row or column axis is a `MultiIndex` is refused,
   since a catalog is a flat table,
   and so are the columns of a `pandas` table whose names collide
   once converted to strings, e.g. the integer `1` and the string `"1"`;
2. a property given as an unordered collection, e.g. a `set`, is refused,
   since the order of the rows it would define is arbitrary,
   and so is a property given as a nested table;
3. every string is stripped of its leading and trailing whitespace,
   in the properties and in the labels,
   and a byte-string label is decoded from UTF-8;
4. every property, and the labels, must be one-dimensional,
   so a property or labels built from a nested sequence are refused;
5. an input describing no row at all is refused;
   an empty property beside a property that holds a value
   is dropped by the next step instead;
6. a property holding no value is dropped and named in a log warning —
   a property holds no value when it is empty,
   or when each of its elements is missing or an empty string;
   a property of zeros holds values and is kept;
7. every property, and the labels, must hold the same number of elements;
8. a property given as a scalar is repeated over the rows;
9. at least one property must be left;
10. the arrays are copied and made read-only.

The errors of the first four steps are reported together,
in a single exception,
so that an input can be fixed in one pass.

The catalog **cannot be changed** once the variable is built:
neither through the table passed to the constructor,
nor through the table that `variable.catalog.to_dataframe()` hands back,
nor through the read-only properties that
`variable.catalog.properties` hands out.
Changing a catalog means declaring a new variable.

!!! warning
    A catalog variable is **not yet solvable**:
    every optimizer and every DOE algorithm rejects a problem including such a variable,
    and unlike a discrete variable,
    no setting relaxes it to a float variable.
    A driver handles catalog variables
    only when it declares it (`handle_catalog_variables`),
    which no GEMSEO driver does yet,
    and which a plugin driver does not by default.

!!! note
    A design space holding a catalog variable
    cannot be exported to CSV,
    since a catalog does not fit in a CSV cell;
    [to_hdf()][gemseo.space.design.DesignSpace.to_hdf] writes it,
    and [from_hdf()][gemseo.space.design.DesignSpace.from_hdf] reads it back.
    Appending to an HDF file, or loading one,
    requires the catalogs to be those stored in the file,
    since the stored values are positions in them;
    otherwise a `ValueError` is raised and the file is left untouched.
    This holds for a design space, e.g. `to_hdf(append=True)`,
    and for a database, e.g. `OptimizationProblem.to_hdf(append=True)`,
    an optimization history backup
    or `Database.update_from_hdf`.

!!! note
    A catalog with a single row gives a variable
    with a single admissible value,
    hence no choice at all.
    Such a variable is accepted as is,
    and still counts as a design variable.

!!! tutorial
    - [Tutorial - Choose among alternatives with catalog variables][]

??? abstract "API"

    - [add_catalog_variable()][gemseo.space.design.DesignSpace.add_catalog_variable]
    - [variables][gemseo.space.design.DesignSpace.variables]
    - [has_variables_of_type()][gemseo.space.variables_view.VariablesView.has_variables_of_type]
    - [CatalogVariable][gemseo.space.variable.CatalogVariable]
    - [Catalog][gemseo.space.catalog.catalog.Catalog]
    - [Catalog.get_position()][gemseo.space.catalog.catalog.Catalog.get_position]

## Integer relaxation { #concept-integer-relaxation }

Some algorithms only support real variables.
In that case,
the design space can relax integer variables by treating them as reals.

## Normalization of the variables { #concept-normalization-of-the-variables }

Optimization algorithms often work better when variables share a comparable scale,
that is why the design space can normalize bounded real variables $x$
into $x_{\mathrm{normalized}}$ in $[0, 1]$:

- $x_{\mathrm{normalized}} = \frac{x-l_b(x)}{u_b(x)-l_b(x)}$,
- $x_{\mathrm{normalized}} = \frac{x}{u_b(x)-l_b(x)}$,

where $l_b(x)$ and $u_b(x)$ are the lower and upper bounds of the variable $x$.

!!! warning
    Discrete and catalog variables cannot be normalized.
    An integer or categorical variable is not normalized either,
    unless [enable_integer_variables_normalization][gemseo.space.design.DesignSpace.enable_integer_variables_normalization]
    is set to `True`,
    in which case an integer variable follows the same formula as a float variable
    and a categorical one is mapped as described in [Samplers (DOE)][concept-samplers-doe].

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
