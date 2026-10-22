---
description: "Migrating doe from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# DOE

The changes of GEMSEO 7 about design of experiments libraries and their settings that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Silent changes

The code runs after bump-gemseo, without error nor comment, and gives other results.

### The unit hypercube variable is no longer named `"x"`

In GEMSEO 6.3, sampling a unit hypercube by dimension, e.g. with `gemseo.compute_doe`, named its single variable `"x"`, so that `reverse=["x"]` reversed all the components with `DiagonalDOE`. In GEMSEO 7, this variable is no longer named `"x"`, and a DOE setting referring to variable names, such as the `reverse` setting of `DiagonalDOE`, has no effect there.

This code runs without error after bump-gemseo, but gives other samples, since the setting is silently ignored. Check the DOE settings referring to variable names when sampling by dimension, and use component indices instead, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo import compute_doe
    samples = compute_doe(2, algo_name="DiagonalDOE", n_samples=3, reverse=["x"])
    print(samples)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo import compute_doe
    from gemseo.doe import DiagonalDOE_Settings
    samples = compute_doe(2, settings_model=DiagonalDOE_Settings(n_samples=3, reverse=["0", "1"]))
    print(samples)
    ```

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### The `random_state` option of `PYDOE_LHS` renamed `seed`

In GEMSEO 7, the `random_state` option of `PYDOE_LHS` has been renamed `seed` and now accepts a NumPy generator too. Besides, the PyDOE algorithms library now relies on the maintained `pydoe` package instead of the unmaintained `pyDOE3`.

bump-gemseo leaves no comment, and the former option is rejected by `PYDOE_LHS_Settings` with a `ValidationError` reading "Extra inputs are not permitted". Pass `seed` instead, e.g. in a `PYDOE_LHS_Settings`, as in the example below. Note that the samples may differ from those of GEMSEO 6.3 for the same seed, since the underlying package changed.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ValidationError: 1 validation error for PYDOE_LHS_Settings random_state Extra inputs are not permitted [type=extra_forbidden, input_value=3, input_type=int] For further information visit https://errors.pydantic.dev/2.14/`
    - **GEMSEO 7 cannot give the GEMSEO 6.3 value here**: the pyDOE algorithms rely on pydoe instead of pyDOE3, whose samples differ for the same seed.
    - **See** issue [#1862](https://gitlab.com/gemseo/dev/gemseo/-/issues/1862), merge request [!2646](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2646).

=== "GEMSEO 6.3"

    ```python
    from gemseo import compute_doe
    samples = compute_doe(2, algo_name="PYDOE_LHS", n_samples=4, random_state=3)
    print(samples)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo import compute_doe
    from gemseo.doe import PYDOE_LHS_Settings
    samples = compute_doe(2, settings_model=PYDOE_LHS_Settings(n_samples=4, seed=3))
    print(samples)
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### `BaseDOELibrary.compute_doe` replaced by `sample_space`

In GEMSEO 7, the `compute_doe` method of `BaseDOELibrary` was removed and replaced by the `sample_space` and `sample_unit_hypercube` methods. The arguments of `sample_space` are renamed: `variables_space` becomes `space`, `unit_sampling` becomes `use_unit_samples` and `settings_model` becomes `settings`.

When the library is only known as a `BaseAlgorithmLibrary`, e.g. when it is created by `DOELibraryFactory`, bump-gemseo cannot rewrite the call and marks it with the comment "cannot transform: library is only known as BaseAlgorithmLibrary; if it is an instance of BaseDOELibrary, compute_doe became sample_space"; left as is, the call raises an `AttributeError`. Replace `compute_doe` by `sample_space` and rename its arguments, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: library is only known as BaseAlgorithmLibrary; if it is an instance of BaseDOELibrary, compute_doe became sample_space`
    - **Error with GEMSEO 7**: `AttributeError: 'PyDOELibrary' object has no attribute 'compute_doe'`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2348](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2348).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.factory import DOELibraryFactory
    from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=2.0)
    space.add_variable("b", lower_bound=-1.0, upper_bound=1.0)
    library = DOELibraryFactory().create("PYDOE_FULLFACT")
    samples = library.compute_doe(variables_space=space, unit_sampling=True, settings_model=PYDOE_FULLFACT_Settings(n_samples=4))
    print(samples)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import PYDOE_FULLFACT_Settings
    from gemseo.doe.factory import DOELibraryFactory
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=2.0)
    space.add_variable("b", lower_bound=-1.0, upper_bound=1.0)
    library = DOELibraryFactory().create("PYDOE_FULLFACT")
    samples = library.sample_space(space=space, use_unit_samples=True, settings=PYDOE_FULLFACT_Settings(n_samples=4))
    print(samples)
    ```

### The settings of `sample_space` are a settings model

In GEMSEO 7, `compute_doe` of `BaseDOELibrary` is replaced by `sample_space`, and the settings of the algorithm, passed as keyword arguments to `compute_doe` in GEMSEO 6.3, become a settings model of the algorithm of the library, e.g. `PYDOE_FULLFACT_Settings(n_samples=3)`, passed as the `settings` argument.

When the library is only known as a `BaseAlgorithmLibrary`, bump-gemseo cannot rewrite the call and marks it with the comment "cannot transform: library is only known as BaseAlgorithmLibrary; if it is an instance of BaseDOELibrary, compute_doe became sample_space"; left as is, the call raises an `AttributeError`. Replace `compute_doe` by `sample_space`, and gather the keyword arguments in the settings model named after the algorithm, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: library is only known as BaseAlgorithmLibrary; if it is an instance of BaseDOELibrary, compute_doe became sample_space`
    - **Error with GEMSEO 7**: `AttributeError: 'PyDOELibrary' object has no attribute 'compute_doe'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.factory import DOELibraryFactory
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=2.0)
    library = DOELibraryFactory().create("PYDOE_FULLFACT")
    print(library.compute_doe(space, n_samples=3))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import PYDOE_FULLFACT_Settings
    from gemseo.doe.factory import DOELibraryFactory
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=2.0)
    library = DOELibraryFactory().create("PYDOE_FULLFACT")
    print(library.sample_space(space, settings=PYDOE_FULLFACT_Settings(n_samples=3)))
    ```

### The `doe_algo_name` field of `MorrisDOE_Settings` removed

In GEMSEO 7, the `doe_algo_name` field of `MorrisDOE_Settings` was removed; use its `doe_algo_settings` field instead, whose type is `BaseDOESettings`. The DOE algorithm is thus given by its settings model, e.g. `PYDOE_FULLFACT_Settings()`, instead of its name.

bump-gemseo marks the use of the field with the comment "MorrisDOE_Settings.doe_algo_name was removed; see the GEMSEO 7 changelog."; left as is, reading `settings.doe_algo_name` raises an `AttributeError`. Pass the settings model of the DOE algorithm as `doe_algo_settings`, and read the algorithm from it when needed, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `MorrisDOE_Settings.doe_algo_name was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'MorrisDOE_Settings' object has no attribute 'doe_algo_name'`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2348](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2348).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.doe.morris_doe.settings.morris_doe_settings import MorrisDOE_Settings
    settings = MorrisDOE_Settings(doe_algo_name="PYDOE_FULLFACT", n_samples=16)
    print([settings.doe_algo_name, settings.n_samples])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import MorrisDOE_Settings
    from gemseo.doe import PYDOE_FULLFACT_Settings
    settings = MorrisDOE_Settings(doe_algo_settings=PYDOE_FULLFACT_Settings(), n_samples=16)
    print([settings.doe_algo_settings.__class__.__name__.removesuffix("_Settings"), settings.n_samples])
    ```

## Deprecated names

The code still works with GEMSEO 7, with a deprecation warning.

### Pass a settings model to `sample_disciplines`

In GEMSEO 6.3, the DOE algorithm of `sample_disciplines` was given by `algo_name` and its settings by keyword arguments. In GEMSEO 7, it can be given by a settings model, e.g. `PYDOE_FULLFACT_Settings(n_samples=5)`, passed as `algo_settings_model`.

bump-gemseo leaves these loose settings as they are, without comment; GEMSEO 7 still runs them with the same values, without warning, but they are not converted to a settings model. Replace `algo_name` and the keyword arguments by the settings model of the algorithm, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: it works, with a deprecation warning.

=== "GEMSEO 6.3"

    ```python
    from gemseo import sample_disciplines
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    dataset = sample_disciplines([AnalyticDiscipline({"y": "2*x"})], space, "y", algo_name="PYDOE_FULLFACT", n_samples=5)
    print(sorted(dataset.get_view(variable_names="y").to_numpy().ravel().tolist()))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo import sample_disciplines
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.doe import PYDOE_FULLFACT_Settings
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    dataset = sample_disciplines([AnalyticDiscipline({"y": "2*x"})], space, "y", algo_settings_model=PYDOE_FULLFACT_Settings(n_samples=5))
    print(sorted(dataset.get_view(variable_names="y").to_numpy().ravel().tolist()))
    ```
