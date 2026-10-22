---
description: "Migrating spaces from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Spaces

The changes of GEMSEO 7 about design spaces and random spaces that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Silent changes

The code runs after bump-gemseo, without error nor comment, and gives other results.

### `get_current_value` returns copies

In GEMSEO 6.3, `DesignSpace.get_current_value` handed out the arrays that the design space keeps, so that `space.get_current_value()[0] = 0.0` changed the value read as an array while leaving the value read as a dictionary unchanged. In GEMSEO 7, it returns copies, so mutating them no longer makes the design space inconsistent.

This code runs without error after bump-gemseo, but mutating the returned array no longer changes the current value of the design space. Check the code mutating the result of `get_current_value` and pass the modified value to `set_current_value`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801), merge request [!2501](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2501).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=5.0, value=1.0)
    current = space.get_current_value()
    current[0] = 4.0
    print(space.get_current_value())
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=5.0, value=1.0)
    current = space.get_current_value()
    current[0] = 4.0
    space.set_current_value(current)
    print(space.get_current_value())
    ```

### `project_into_bounds` may return a float array

In GEMSEO 7, `DesignSpace.project_into_bounds` no longer preserves the dtype of its `x_vect` argument: an integer input is promoted to float whenever the bounds are float (any infinite bound, or a mix of real and integer variables). The result is numerically more correct, as float bounds are no longer truncated to fit an integer array.

This code runs without error after bump-gemseo, but returns a float array where GEMSEO 6.3 returned an integer one. Check the code relying on an integer result and cast it explicitly, e.g. with `.astype(int)`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801), merge request [!2501](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2501).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=np.inf)
    projected = space.project_into_bounds(np.array([-3, 4]))
    print([projected.dtype.kind, projected])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=np.inf)
    projected = space.project_into_bounds(np.array([-3, 4])).astype(int)
    print([projected.dtype.kind, projected])
    ```

### The design space copies and casts the values it stores

In GEMSEO 7, `DesignSpace.set_current_value`, `DesignSpace.set_current_variable` and `DesignSpace.add_variable` store a copy of the value, cast to `float64` for a real variable unless it is complex, and the finite bounds of a variable are stored with the NumPy type of its components. In GEMSEO 6.3, the design space stored the array of the caller and kept its NumPy type, e.g. `int64`, for the value as for the bounds.

This code runs without error after bump-gemseo, but gives other results: mutating an array after passing it no longer changes the current value, and the value and the bounds of a real variable are float arrays. Check the code doing so: set the value after modifying the array, and cast explicitly, e.g. with `.astype(int)`, where an integer array is expected, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1845](https://gitlab.com/gemseo/dev/gemseo/-/issues/1845), merge request [!2629](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2629).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=[0, 0], upper_bound=[5, 5], value=0.0)
    value = np.array([1, 2])
    space.set_current_value({"x": value})
    value[0] = 3
    current = space.get_current_value(as_dict=True)["x"]
    print([current.dtype.kind, current, space.get_lower_bound("x").dtype.kind])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=[0, 0], upper_bound=[5, 5], value=0.0)
    value = np.array([1, 2])
    value[0] = 3
    space.set_current_value({"x": value})
    current = space.get_current_value(as_dict=True)["x"].astype(int)
    print([current.dtype.kind, current, space.variables["x"].lower_bound.astype(int).dtype.kind])
    ```

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### The serialization constants of `DesignSpace` removed

The class attributes of `DesignSpace` holding the field names of the HDF5 and CSV serialization were removed in GEMSEO 7 without a public replacement, namely `DESIGN_SPACE_GROUP`, `NAME_GROUP`, `NAMES_GROUP`, `LB_GROUP`, `UB_GROUP`, `VAR_TYPE_GROUP`, `VALUE_GROUP`, `SIZE_GROUP`, `MINIMAL_FIELDS` and `TABLE_NAMES`. They were implementation details of `to_hdf`, `from_hdf`, `to_csv`, `from_csv` and `get_pretty_table`.

bump-gemseo replaces `DesignSpace.DESIGN_SPACE_GROUP` by its value `"design_space"`, but leaves no comment on the other constants, whose use raises an `AttributeError`, e.g. `type object 'DesignSpace' has no attribute 'NAME_GROUP'`. Remove the code relying on them.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: type object 'DesignSpace' has no attribute 'NAME_GROUP'`
    - **See** issue [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801), merge request [!2501](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2501).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    print([DesignSpace.DESIGN_SPACE_GROUP, DesignSpace.NAME_GROUP, DesignSpace.NAMES_GROUP, DesignSpace.LB_GROUP, DesignSpace.UB_GROUP, DesignSpace.VAR_TYPE_GROUP, DesignSpace.VALUE_GROUP, DesignSpace.SIZE_GROUP, list(DesignSpace.MINIMAL_FIELDS), list(DesignSpace.TABLE_NAMES)])
    ```

=== "GEMSEO 7"

    ```python
    print("<removed in GEMSEO 7>")
    ```

### The extra options of the `DesignSpace` file methods removed

In GEMSEO 7, the `**options` arguments of `DesignSpace.from_file` were removed because they were deprecated and ignored, those of `DesignSpace.to_file` because they were deprecated (the new `append` argument is available instead), and the `header_char` and `**table_options` arguments of `DesignSpace.to_csv` because they were deprecated.

bump-gemseo leaves no comment on these calls, which raise a `TypeError` for the unexpected keyword argument, e.g. `DesignSpace.to_csv() got an unexpected keyword argument 'sortby'`. Remove these keyword arguments, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: DesignSpace.to_csv() got an unexpected keyword argument 'sortby'`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2303](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2303).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("b", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("a", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    space.to_csv("space.csv", sortby="name")
    space.to_file("space.txt", header_char="")
    loaded = DesignSpace.from_file("space.txt", header_char="")
    print([sorted(loaded.variable_names), loaded.get_current_value(as_dict=True)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("b", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("a", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    space.to_csv("space.csv")
    space.to_file("space.txt")
    loaded = DesignSpace.from_file("space.txt")
    print([sorted(loaded.variables), loaded.get_current_value(as_dict=True)])
    ```

### Stricter bounds of an integer variable

In GEMSEO 6.3, a non-integer scalar bound of an integer variable was truncated, and a finite bound beyond the range of a 64-bit integer was accepted and stored as a floating-point number. In GEMSEO 7, an integer variable rejects a non-integer scalar bound, and an `IntegerVariable` rejects a finite bound outside the range of a 64-bit integer; an infinite bound is still accepted.

bump-gemseo leaves no comment, and `add_variable` raises a `ValidationError`, e.g. `The following lower bound component is neither integer nor infinite while the variable is of type integer: 2.5 (index 0).` Give integer bounds (the truncated value, to keep the 6.3 behaviour) and replace a huge finite bound by an infinite one such as `np.inf`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ValidationError: 1 validation error for IntegerVariable Value error, The following lower bound component is neither integer nor infinite while the variable is of type integer: 2.5 (index 0). [type=value_error, input_valu`
    - **See** issue [#1845](https://gitlab.com/gemseo/dev/gemseo/-/issues/1845), merge request [!2629](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2629); merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("i", type_="integer", lower_bound=2.5, upper_bound=9, value=3)
    space.add_variable("n", type_="integer", lower_bound=0, upper_bound=np.array([1e30]), value=1)
    print([space.get_lower_bound("i"), space.get_upper_bound("n") > 1e20])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("i", type_="integer", lower_bound=2, upper_bound=9, value=3)
    space.add_variable("n", type_="integer", lower_bound=0, upper_bound=np.inf, value=1)
    print([space.variables["i"].lower_bound, space.variables["n"].upper_bound > 1e20])
    ```

### The metaclass of `DesignSpace` is `ABCGoogleDocstringInheritanceMeta`

In GEMSEO 7, `DesignSpace` uses the `ABCGoogleDocstringInheritanceMeta` metaclass instead of plain `type`, so that a subclass inherits the docstrings of the members it overrides. A subclass combining `DesignSpace` with another metaclass must use one compatible with it, such as `ABCMeta`.

bump-gemseo leaves no comment, and defining such a subclass raises `TypeError: metaclass conflict`. Derive the other metaclass from `abc.ABCMeta` and give the subclass a metaclass deriving from both `type(DesignSpace)` and this metaclass, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases`
    - **See** issue [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801), merge request [!2501](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2501).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace


    class Meta(type):
        pass


    class Tagged(metaclass=Meta):
        tag = "tagged"


    class TaggedSpace(DesignSpace, Tagged):
        pass


    space = TaggedSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    print([space.tag, space.dimension])
    ```

=== "GEMSEO 7"

    ```python
    import abc
    from gemseo.space import DesignSpace


    class Meta(abc.ABCMeta):
        pass


    class Tagged(metaclass=Meta):
        tag = "tagged"


    class TaggedSpace(DesignSpace, Tagged, metaclass=type("TaggedSpaceMeta", (type(DesignSpace), Meta), {})):
        pass


    space = TaggedSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    print([space.tag, space.dimension])
    ```

### The `simplify` argument of `get_pretty_table` removed

The `simplify` argument of `DesignSpace.get_pretty_table` was removed in GEMSEO 7, together with the `ParameterSpace` implementation honoring it, which dropped the bound, value and type columns from the tabular view of a space of uncertain variables only. The tabular view of a `RandomSpace` has a name column and a distribution column, so there is nothing left to simplify.

bump-gemseo leaves no comment, and the call raises `TypeError: DesignSpace.get_pretty_table() got an unexpected keyword argument 'simplify'`. Remove the argument, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: DesignSpace.get_pretty_table() got an unexpected keyword argument 'simplify'`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    table = space.get_pretty_table(simplify=False)
    print(["x" in str(table), len(table.rows)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    table = space.get_pretty_table()
    print(["x" in str(table), len(table.rows)])
    ```

### Read-only dimension, bounds and normalization masks of a `DesignSpace`

In GEMSEO 7, `DesignSpace.dimension` is a read-only property, `DesignSpace.normalize` (renamed `DesignSpace.name_to_normalization_mask`, the former name being a deprecated alias) and `names_to_indices` are read-only mappings, the normalization masks are read-only arrays, and `DesignSpace.get_lower_bounds` and `DesignSpace.get_upper_bounds` return read-only arrays. Assigning these attributes or items, or mutating these arrays in place, raises an error.

bump-gemseo leaves no comment on such code, which raises e.g. `TypeError: 'NormalizationMasks' object does not support item assignment`. Do not assign `dimension`; copy the bounds before changing them, e.g. `space.get_lower_bounds().copy()`; and, since the masks of the space can no longer be changed, work on copies of the masks of `name_to_normalization_mask`, as in the example below, which applies the modified masks itself.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: 'NormalizationMasks' object does not support item assignment`
    - **See** issue [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801), merge request [!2501](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2501); issue [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801), merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=2.0, value=1.0)
    space.add_variable("y", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    space.normalize["y"] = np.array([False])
    space.normalize["x"][1] = False
    lower = space.get_lower_bounds()
    lower[0] = -5.0
    space.dimension = space.dimension
    print([space.normalize_vect(np.array([1.0, 1.0, 0.5])), lower, space.get_lower_bounds()])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=2.0, value=1.0)
    space.add_variable("y", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    masks = {name: mask.copy() for name, mask in space.name_to_normalization_mask.items()}
    masks["y"] = np.array([False])
    masks["x"][1] = False
    mask = space.convert_dict_to_array(masks).astype(bool)
    x_vect = np.array([1.0, 1.0, 0.5])
    lower = space.get_lower_bounds().copy()
    upper = space.get_upper_bounds()
    normalized = np.where(mask, (x_vect - lower) / (upper - lower), x_vect)
    lower[0] = -5.0
    print([normalized, lower, space.get_lower_bounds()])
    ```

### An unknown variable type raises a `ValueError`

In GEMSEO 7, `DesignSpace.add_variable` raises a `ValueError` instead of a `pydantic.ValidationError` when the type of the variable is unknown.

bump-gemseo leaves no comment, and code catching `ValidationError` lets the error through, e.g. `ValueError: Only real and integer variables may be declared through the type_ argument of add_variable; use the add_<kind>_variable method instead.` Catch `ValueError` instead, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ValueError: Only real and integer variables may be declared through the type_ argument of add_variable; use the add_<kind>_variable method instead.`
    - **See** issue [#1845](https://gitlab.com/gemseo/dev/gemseo/-/issues/1845), merge request [!2629](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2629).

=== "GEMSEO 6.3"

    ```python
    from pydantic import ValidationError
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    try:
        space.add_variable("x", type_="complex")
        message = "accepted"
    except ValidationError:
        message = "rejected"
    print([message, len(space.variable_names)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.space import DesignSpace
    space = DesignSpace()
    try:
        space.add_variable("x", type_="complex")
        message = "accepted"
    except ValueError:
        message = "rejected"
    print([message, len(space.variables)])
    ```

### The `x_vect` argument of the `DesignSpace` methods

In GEMSEO 7, the concatenated design-vector argument is uniformly named `x_vect`. It was named `x_vec` in `DesignSpace.get_active_bounds`, `x_c` in `DesignSpace.project_into_bounds`, `vector` in `DesignSpace.transform_vect` and `DesignSpace.untransform_vect`, and `x_array` in `DesignSpace.convert_array_to_dict`.

bump-gemseo leaves no comment on these calls; a call passing the former name by keyword raises a `TypeError`, e.g. `DesignSpace.get_active_bounds() got an unexpected keyword argument 'x_vec'`. Rename the keyword to `x_vect`, as in the example below; positional calls are unchanged.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: DesignSpace.get_active_bounds() got an unexpected keyword argument 'x_vec'`
    - **See** issue [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801), merge request [!2501](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2501).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=2.0, value=1.0)
    lower, upper = space.get_active_bounds(x_vec=np.array([0.0, 1.0]))
    projected = space.project_into_bounds(x_c=np.array([-1.0, 3.0]))
    transformed = space.transform_vect(vector=np.array([1.0, 2.0]))
    restored = space.untransform_vect(vector=transformed)
    as_dict = space.convert_array_to_dict(x_array=np.array([0.5, 1.5]))
    print([lower["x"], upper["x"], projected, transformed, restored, as_dict["x"]])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=2.0, value=1.0)
    lower, upper = space.get_active_bounds(x_vect=np.array([0.0, 1.0]))
    projected = space.project_into_bounds(x_vect=np.array([-1.0, 3.0]))
    transformed = space.transform_vect(x_vect=np.array([1.0, 2.0]))
    restored = space.untransform_vect(x_vect=transformed)
    as_dict = space.convert_array_to_dict(x_vect=np.array([0.5, 1.5]))
    print([lower["x"], upper["x"], projected, transformed, restored, as_dict["x"]])
    ```

### `IshigamiSpace` is created by `random_space_factory`

In GEMSEO 6.3, `IshigamiSpace` and `WingWeightUncertainSpace` were design spaces found by `DesignSpaceFactory`. In GEMSEO 7, they derive from `RandomSpace` (the latter being renamed `WingWeightRandomSpace`) and they are found by the new `random_space_factory` instead of `design_space_factory`.

bump-gemseo leaves no comment, and creating `"IshigamiSpace"` with the design space factory raises an `ImportError` listing the available design spaces. Import `random_space_factory` from `gemseo.space.factory` and call its `create` method, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ImportError: The class IshigamiSpace is not available; the available ones are: AerostructureDesignSpace, ScalableDesignSpace, SellarDesignSpace and SobieskiDesignSpace.`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space_factory import DesignSpaceFactory
    space = DesignSpaceFactory().create("IshigamiSpace")
    print([space.dimension, "IshigamiSpace" in DesignSpaceFactory().class_names])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.space.factory import random_space_factory
    space = random_space_factory.create("IshigamiSpace")
    print([space.dimension, "IshigamiSpace" in random_space_factory.class_names])
    ```

### `compute_samples(as_dict=True)` returns a dictionary of arrays

In GEMSEO 6.3, `ParameterSpace.compute_samples` with `as_dict=True` returned a list of dictionaries, one per sample. In GEMSEO 7, `RandomSpace.compute_samples` with `as_dict=True` returns a dictionary mapping the name of a random variable to an array shaped as `(n_samples, size)`.

bump-gemseo leaves no comment, and indexing the result by sample raises `KeyError: 0`. Index the arrays of the dictionary by sample instead, or rebuild the list of dictionaries from them, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `KeyError: 0`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
    samples = IshigamiSpace().compute_samples(3, as_dict=True)
    print([len(samples), sorted(samples[0]), [len(sample["x1"]) for sample in samples]])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.problem.uncertainty.ishigami import IshigamiSpace
    samples = IshigamiSpace().compute_samples(3, as_dict=True)
    as_list = [{name: value[index] for name, value in samples.items()} for index in range(3)]
    print([len(as_list), sorted(as_list[0]), [len(sample["x1"]) for sample in as_list]])
    ```

### The table options of `write_design_space` removed

The `header_char` and `**table_options` arguments of `write_design_space` were removed in GEMSEO 7 because they were deprecated; the new argument `append` is available instead.

bump-gemseo leaves no comment on such a call, which raises a `TypeError` for the unexpected keyword argument, e.g. `sortby`. Remove these keyword arguments, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: write_design_space() got an unexpected keyword argument 'sortby'`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2303](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2303).

=== "GEMSEO 6.3"

    ```python
    from gemseo import read_design_space
    from gemseo import write_design_space
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("b", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("a", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    write_design_space(space, "space.txt", sortby="name")
    print(sorted(read_design_space("space.txt").variable_names))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo import read_design_space
    from gemseo import write_design_space
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("b", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("a", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    write_design_space(space, "space.txt")
    print(sorted(read_design_space("space.txt").variables))
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### `DesignSpace.DesignVariableType` replaced by `VariableType`

The class attribute `DesignSpace.DesignVariableType` is removed in GEMSEO 7; the enumeration of the variable types is now available as `gemseo.enum.VariableType`. Its member `FLOAT` is deprecated in favour of `VariableType.REAL`, and the type of a real variable is named `"real"` instead of `"float"`, so comparing a variable type to `"float"` is now `False`.

bump-gemseo rewrites `DataType.FLOAT` and `get_type(name)` automatically, but marks `DesignSpace.DesignVariableType` with the comment "use gemseo.enum.VariableType instead"; left as is, the line raises an `AttributeError`. Import `VariableType` from `gemseo.enum`, use `VariableType.REAL`, and read the type of a variable with `space.variables[name].type`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `use gemseo.enum.VariableType instead`
    - **Error with GEMSEO 7**: `AttributeError: type object 'DesignSpace' has no attribute 'DesignVariableType'`
    - **GEMSEO 7 cannot give the GEMSEO 6.3 value here**: the type of a real variable is named "real" instead of "float" (changelog 1931).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", type_=DesignSpace.DesignVariableType.FLOAT)
    print([DesignSpace.DesignVariableType.FLOAT, space.get_type("x")])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.enum import VariableType
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", type_=VariableType.REAL)
    print([VariableType.REAL, space.variables["x"].type])
    ```

### An unknown variable name raises a `KeyError`

In GEMSEO 6.3, `get_size`, `get_type`, `set_lower_bound` and other methods of `DesignSpace` raised a `ValueError` for an unknown variable name. In GEMSEO 7, any unknown variable name raises a `KeyError` with the message `No variable named '<name>'.`, and the shortcuts `get_size(name)` and `get_type(name)` were removed in favour of `space.variables[name].size` and `.type`.

bump-gemseo cannot rewrite a method used without its call, e.g. `space.get_size` iterated over, and marks the line with a comment such as "cannot transform: DesignSpace.get_size became {receiver}.variables[{name}].size, which needs the arguments of a call."; left as is, the code raises an `AttributeError`. Access the variable with `space.variables[name]` and catch `KeyError` instead of `ValueError`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: DesignSpace.get_size became {receiver}.variables[{name}].size, which needs the arguments of a call.` / `cannot transform: DesignSpace.get_type became {receiver}.variables[{name}].type, which needs the arguments of a call.`
    - **Error with GEMSEO 7**: `AttributeError: 'DesignSpace' object has no attribute 'get_size'`
    - **See** issue [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801), merge request [!2501](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2501).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    messages = []
    for method in (space.get_size, space.get_type):
        try:
            method("unknown")
        except ValueError:
            messages.append("unknown variable")
    print(messages)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    messages = []
    for name in ("unknown", "unknown"):
        try:
            space.variables[name]
        except KeyError:
            messages.append("unknown variable")
    print(messages)
    ```

### A `RandomSpace` holds no deterministic variable

The class `ParameterSpace`, which mixed deterministic and uncertain variables, was removed in GEMSEO 7: a space is now either a `DesignSpace`, defined by bounds, or a `RandomSpace`, defined by probability distributions. A current value being a notion of the deterministic spaces, `has_current_value`, `get_current_value`, `set_current_value` and the like belong to `DesignSpace` only.

bump-gemseo marks these lines with the comments "a RandomSpace holds only random variables; add a deterministic variable to a DesignSpace instead" and "a current value is a DesignSpace notion; ..."; left as is, `add_variable` raises e.g. `TypeError: RandomSpace.add_variable() got an unexpected keyword argument 'lower_bound'`. Split the space into a `DesignSpace` for the deterministic variables and their current value, and a `RandomSpace` for the random ones, whose reference point, the mean of the distributions, is given by `reference_value`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `a RandomSpace holds only random variables; add a deterministic variable to a DesignSpace instead` / `a current value is a DesignSpace notion; use nominal_value to read the reference point of a space, which is the mean of the distributions here`
    - **Error with GEMSEO 7**: `TypeError: RandomSpace.add_variable() got an unexpected keyword argument 'lower_bound'`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.parameter_space import ParameterSpace
    space = ParameterSpace()
    space.add_variable("w", lower_bound=0.0, upper_bound=10.0)
    space.add_random_variable("u", "SPNormalDistribution", mu=3.0, sigma=1.0)
    missing = space.has_current_value
    space.initialize_missing_current_values()
    space.set_current_variable("w", np.array([4.0]))
    space.set_current_value({"w": np.array([5.0]), "u": np.array([3.0])})
    space.to_complex()
    print([missing, space.has_current_value, space.get_current_value(as_dict=True)])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.space import DesignSpace
    from gemseo.space import RandomSpace
    from gemseo.uncertainty.distribution import SPNormalDistribution_Settings
    design_space = DesignSpace()
    design_space.add_variable("w", lower_bound=0.0, upper_bound=10.0)
    random_space = RandomSpace()
    random_space.add_variable("u", SPNormalDistribution_Settings(mu=3.0, sigma=1.0))
    missing = design_space.has_current_value
    design_space.initialize_missing_current_values()
    design_space.set_current_variable("w", np.array([4.0]))
    design_space.set_current_value({"w": np.array([5.0])})
    design_space.to_complex()
    print([missing, design_space.has_current_value, {"w": design_space.get_current_value(as_dict=True)["w"].real, "u": random_space.reference_value["u"]}])
    ```

### `ParameterSpace.normalize` removed

`ParameterSpace` is replaced by `RandomSpace` in GEMSEO 7, and a `RandomSpace` has no normalization: its only vector mapping is the iso-probabilistic `transform_vect`/`untransform_vect` to and from the unit hypercube. A space can no longer mix deterministic and random variables either, so `ParameterSpace.normalize` has no counterpart.

bump-gemseo marks the line with the comment "ParameterSpace.normalize was removed; see the GEMSEO 7 changelog.", and marks the deterministic variables added to the space with the comment "a RandomSpace holds only random variables; add a deterministic variable to a DesignSpace instead". Remove the use of `normalize`; if you need the normalization masks of deterministic variables, declare them in a `DesignSpace` and read its `name_to_normalization_mask`.

!!! info "After bump-gemseo"

    - **Comment**: `a RandomSpace holds only random variables; add a deterministic variable to a DesignSpace instead` / `ParameterSpace.normalize was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `TypeError: RandomSpace.add_variable() got an unexpected keyword argument 'lower_bound'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.parameter_space import ParameterSpace
    space = ParameterSpace()
    space.add_random_variable("u", "OTNormalDistribution")
    space.add_variable("d", lower_bound=0.0, upper_bound=1.0, value=0.5)
    print({name: mask.tolist() for name, mask in space.normalize.items()})
    ```

=== "GEMSEO 7"

    ```python
    print("<removed in GEMSEO 7>")
    ```

### Random variables are added with settings models

In GEMSEO 7, `ParameterSpace` is replaced by `RandomSpace`, and `ParameterSpace.add_random_variable` and `ParameterSpace.add_random_vector` are replaced by `RandomSpace.add_variable(name, *settings)`. A random variable is defined by the settings of the marginal probability distributions of its components only, e.g. `SPNormalDistribution_Settings(mu=0.0, sigma=1.0)`.

bump-gemseo renames the methods and converts a distribution name with its parameters into a settings model, but marks with a comment what it cannot convert; left as is, the call raises e.g. `TypeError: RandomSpace.add_variable() got an unexpected keyword argument 'size'`. Follow each comment, as in the example below:

- for `size`, repeat the settings model, i.e. `add_variable(name, *[settings] * size)`;
- for `add_random_vector`, pass one settings model per component, by position;
- for `interfaced_distribution` and `interfaced_distribution_parameters`, set `interfaced_distribution` and `parameters` in an `SPDistribution_Settings` or an `OTDistribution_Settings`.

!!! info "After bump-gemseo"

    - **Comment**: `repeat the settings model size times, i.e. add_variable(name, *[settings] * size)` / `pass the distribution settings models, e.g. SPNormalDistribution_Settings(mu=0.0, sigma=1.0), by position, one per component; a distribution name with its parameters must be written as settings models` / `pass one distribution settings model per component, i.e. add_variable(name, *settings)`
    - **Error with GEMSEO 7**: `TypeError: RandomSpace.add_variable() got an unexpected keyword argument 'size'`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.parameter_space import ParameterSpace
    space = ParameterSpace()
    space.add_random_variable("a", "SPNormalDistribution", size=2, mu=1.0, sigma=2.0)
    space.add_random_variable("b", "SPDistribution", interfaced_distribution="expon", interfaced_distribution_parameters={"scale": 2.0})
    space.add_random_vector("c", "SPUniformDistribution", size=2, minimum=[0.0, 1.0], maximum=[1.0, 3.0])
    space.add_random_vector("d", "SPDistribution", interfaced_distribution="expon", interfaced_distribution_parameters={"scale": [0.5]})
    print([space.dimension, space.distribution.mean])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.space import RandomSpace
    from gemseo.uncertainty.distribution import SPNormalDistribution_Settings
    from gemseo.uncertainty.distribution import SPDistribution_Settings
    from gemseo.uncertainty.distribution import SPUniformDistribution_Settings
    space = RandomSpace()
    space.add_variable("a", *[SPNormalDistribution_Settings(mu=1.0, sigma=2.0)] * 2)
    space.add_variable("b", SPDistribution_Settings(interfaced_distribution="expon", parameters={"scale": 2.0}))
    space.add_variable("c", SPUniformDistribution_Settings(minimum=0.0, maximum=1.0), SPUniformDistribution_Settings(minimum=1.0, maximum=3.0))
    space.add_variable("d", SPDistribution_Settings(interfaced_distribution="expon", parameters={"scale": 0.5}))
    print([space.dimension, space.variables.distribution.mean])
    ```

### Distributions and statistics of a `RandomSpace`

In GEMSEO 7, the probabilistic data of a `RandomSpace` is read through `variables`: `variables.distribution` is the joint distribution of the space and `variables[name].distribution` that of a variable. `ParameterSpace.build_joint_distribution` was replaced by `add_copula`, the joint distribution being built on demand. `ParameterSpace.evaluate_cdf` was removed because it did not implement the CDF but the Rosenblatt transformation, and `ParameterSpace.get_tabular_view` was replaced by `get_pretty_table`, whose table has a name column and a distribution column only.

bump-gemseo marks each of these lines with a comment, e.g. "use space.variables[name].distribution instead"; left as is, the code raises an `AttributeError`. Follow each comment, as in the example below: read the distributions from `space.variables`, call `add_copula` only for dependent random variables, use `transform_vect` for the former `evaluate_cdf` (or `compute_cdf` of a distribution for its CDF), and read the statistics from the distribution of a variable, e.g. `space.variables[name].distribution.mean`.

!!! info "After bump-gemseo"

    - **Comment**: `use space.variables[name].distribution instead` / `use add_copula, and read the joint distribution from space.variables` / `use transform_vect and untransform_vect instead`
    - **Error with GEMSEO 7**: `AttributeError: 'IshigamiSpace' object has no attribute 'distributions'`
    - **See** issue [#1717](https://gitlab.com/gemseo/dev/gemseo/-/issues/1717), merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
    space = IshigamiSpace()
    means = {name: distribution.mean for name, distribution in space.distributions.items()}
    space.build_joint_distribution()
    cdf = space.evaluate_cdf({"x1": np.array([0.0]), "x2": np.array([np.pi / 2]), "x3": np.array([-np.pi / 2])})
    table = space.get_tabular_view()
    print([means, space.distribution.dimension, {name: value for name, value in cdf.items()}, "x3" in table])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.problem.uncertainty.ishigami import IshigamiSpace
    space = IshigamiSpace()
    means = {name: space.variables[name].distribution.mean for name in space.variables}
    cdf = space.convert_array_to_dict(space.transform_vect(np.array([0.0, np.pi / 2, -np.pi / 2])))
    table = space.get_pretty_table()
    print([means, space.variables.distribution.dimension, {name: value for name, value in cdf.items()}, "x3" in str(table)])
    ```

### `ScalableDesignSpace` holds no random variable

In GEMSEO 7, a design space no longer holds uncertain variables, so `ScalableDesignSpace.add_random_variable` and `ScalableDesignSpace.add_random_vector` no longer exist and the `add_uncertain_variables` argument of `ScalableDesignSpace` was removed.

bump-gemseo marks these calls with the comment "add the random variables to a separate RandomSpace instead", and the use of `deterministic_variables` with "ParameterSpace.deterministic_variables was removed; see the GEMSEO 7 changelog."; left as is, the code raises an `AttributeError`. Create a `RandomSpace` next to the design space, add the random variables to it with `add_variable` and distribution settings models, and use `space.variables` for the deterministic variables, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `add the random variables to a separate RandomSpace instead` / `ParameterSpace.deterministic_variables was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'ScalableDesignSpace' object has no attribute 'add_random_variable'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.mdo.scalable.parametric.scalable_design_space import ScalableDesignSpace
    space = ScalableDesignSpace()
    space.add_random_variable("u", "SPNormalDistribution", mu=1.0, sigma=0.5)
    space.add_random_vector("v", "SPUniformDistribution", size=2, minimum=[0.0, 0.0], maximum=[1.0, 2.0])
    print([len(space.deterministic_variables), space.distribution.mean])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.problem.mdo.scalable.parametric import ScalableDesignSpace
    from gemseo.space import RandomSpace
    from gemseo.uncertainty.distribution import SPNormalDistribution_Settings
    from gemseo.uncertainty.distribution import SPUniformDistribution_Settings
    space = ScalableDesignSpace()
    random_space = RandomSpace()
    random_space.add_variable("u", SPNormalDistribution_Settings(mu=1.0, sigma=0.5))
    random_space.add_variable("v", SPUniformDistribution_Settings(minimum=0.0, maximum=1.0), SPUniformDistribution_Settings(minimum=0.0, maximum=2.0))
    print([len(space.variables), random_space.variables.distribution.mean])
    ```

### The `normalize_ds` argument of `get_value_and_bounds` removed

In GEMSEO 7, `get_value_and_bounds()` moved from `gemseo.algos.design_space_utils` to `gemseo.space.util`, its `normalize_ds` argument was removed and its `as_dict` argument is keyword-only. It returns the value and the bounds of the space it is passed as they are.

bump-gemseo marks the call with a comment. In user code, call `get_value_and_bounds(space)` and, when the normalized values are needed, normalize them with `space.normalize_vect`, as in the example below. In a driver library, the comment explains what to do: pass `problem.input_space`, the working space, already normalized when `normalize_design_space` is set, map a point back with `self._transformation.inverse_transform_value(x)`, read what refers to the problem of the user from `self._original_problem.input_space`, and set `_iterates_on_working_problem = False` in a library building its problem from the coefficients of the functions, as `ScipyMILP` and `ScipyLinprog` do.

!!! info "After bump-gemseo"

    - **Comment**: `get_value_and_bounds returns the value and the bounds of the space it is passed as they are; in a driver library, problem.input_space is the working space, already normalized when normalize_design_space is set, so pass it and map a point back to the coordinates the user declared with self._transformation.inverse_transform_value(x), always, rather than with unnormalize_vect guarded by normalize_design_space, which would map it twice (see gemseo.space.transformation); the other way round, the problem a library is given (self._problem, and the argument of _pre_run and _run) is the working problem, so what refers to the problem of the user, such as points the user gives (an initial design as an array or a dataset, the database) and the flags of its space read in _pre_run (enable_integer_variables_normalization), must be read from self._original_problem.input_space; a library that builds its problem from the coefficients and the bounds of the functions, e.g. a linear one reading their coefficients, which are not normalized, must instead set the class attribute _iterates_on_working_problem = False, as ScipyMILP and ScipyLinprog do`
    - **With GEMSEO 7**: no error, but other results.

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.design_space_utils import get_value_and_bounds
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=4.0, value=1.0)
    print([get_value_and_bounds(space, True), get_value_and_bounds(space, False)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.space import DesignSpace
    from gemseo.space.util import get_value_and_bounds
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=4.0, value=1.0)
    value, lower_bound, upper_bound = get_value_and_bounds(space)
    normalized = (space.normalize_vect(value), space.normalize_vect(lower_bound), space.normalize_vect(upper_bound))
    print([normalized, (value, lower_bound, upper_bound)])
    ```

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. Unless an entry above gives an equivalent, they have no replacement: remove the code that uses them.

The methods of `ParameterSpace` below were removed with it: GEMSEO 7 has `RandomSpace`, which holds random variables only; the deterministic variables, their bounds and their normalization belong to a `DesignSpace`.

- `ParameterSpace.add_variables_from`
- `ParameterSpace.check_membership`
- `ParameterSpace.deterministic_variables`
- `ParameterSpace.enable_integer_variables_normalization`
- `ParameterSpace.extend`
- `ParameterSpace.extract_deterministic_space`
- `ParameterSpace.extract_uncertain_space`
- `ParameterSpace.from_csv`
- `ParameterSpace.from_file`
- `ParameterSpace.from_hdf`
- `ParameterSpace.get_active_bounds`
- `ParameterSpace.get_lower_bound`
- `ParameterSpace.get_lower_bounds`
- `ParameterSpace.get_upper_bound`
- `ParameterSpace.get_upper_bounds`
- `ParameterSpace.init_from_dataset`
- `ParameterSpace.is_deterministic`
- `ParameterSpace.is_uncertain`
- `ParameterSpace.normalize_grad`
- `ParameterSpace.normalize_vect`
- `ParameterSpace.project_into_bounds`
- `ParameterSpace.round_vect`
- `ParameterSpace.set_lower_bound`
- `ParameterSpace.set_upper_bound`
- `ParameterSpace.to_csv`
- `ParameterSpace.to_design_space`
- `ParameterSpace.to_file`
- `ParameterSpace.to_hdf`
- `ParameterSpace.to_scalar_variables`
- `ParameterSpace.unnormalize_grad`
- `ParameterSpace.unnormalize_vect`
