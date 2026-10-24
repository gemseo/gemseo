---
status: new
description: "Migrating utilities from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Utilities

The changes of GEMSEO 7 about configuration, logging, testing helpers and other utilities that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Silent changes

The code runs after bump-gemseo, without error or comment, and gives other results.

### `epsilon` is a builtin `float` and no longer upcasts `float32` arrays

The `gemseo.utils.derivatives.error_estimators.EPSILON` constant was renamed to `gemseo.util.constant.epsilon`, which is now a builtin `float` instead of a `numpy.float64`, so that it no longer upcasts `float32` arrays.
Once the import is updated, the code runs without error, but an operation between a `float32` array and `epsilon` now gives a `float32` array instead of a `float64` one.

Check the computations mixing `epsilon` with `float32` arrays; if you need the GEMSEO 6.3 precision, convert the constant with `np.float64(epsilon)`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1873](https://gitlab.com/gemseo/dev/gemseo/-/issues/1873), merge request [!2698](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2698).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.utils.derivatives.error_estimators import EPSILON
    values = np.ones(2, dtype=np.float32) * EPSILON
    print([values.dtype.name, type(EPSILON).__name__.replace("float64", "float")])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.util.constant import epsilon
    values = np.ones(2, dtype=np.float32) * np.float64(epsilon)
    print([values.dtype.name, type(epsilon).__name__])
    ```

### `get_sub_disciplines` replaces any object with a `disciplines` attribute by its disciplines

In GEMSEO 6.3, `get_sub_disciplines` replaced only scenarios, formulations and `ProcessDiscipline` objects by the disciplines they are made of, and kept any other object as is.
In GEMSEO 7, it replaces any object with a non-empty `disciplines` attribute, e.g. a discipline of your own that stores other disciplines in such an attribute.
bump-gemseo only renames the module, `gemseo.utils.discipline` into `gemseo.util.discipline`: the code runs without error, but such an object is replaced by its disciplines.

If such an object must be kept as a whole, store its disciplines in an attribute with another name, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673).

=== "GEMSEO 6.3"

    ```python
    from gemseo.core.discipline import Discipline
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.utils.discipline import get_sub_disciplines

    class Wrapper(Discipline):
        def __init__(self, disciplines):
            super().__init__()
            self.disciplines = disciplines

        def _run(self, input_data):
            return None

    wrapper = Wrapper([AnalyticDiscipline({"y": "x"}, name="inner")])
    print([discipline.name for discipline in get_sub_disciplines([wrapper])])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.discipline import Discipline
    from gemseo.util.discipline import get_sub_disciplines

    class Wrapper(Discipline):
        def __init__(self, disciplines):
            super().__init__()
            self.wrapped_disciplines = disciplines

        def _run(self, input_data):
            return None

    wrapper = Wrapper([AnalyticDiscipline({"y": "x"}, name="inner")])
    print([discipline.name for discipline in get_sub_disciplines([wrapper])])
    ```

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### `filter_names` no longer accepts `None`

The `names_to_keep` argument of `filter_names` no longer accepts `None` to keep all the names; an empty iterable does it.
This argument can now also be an iterator, e.g. a generator, which used to keep no name.
bump-gemseo only renames the module, `gemseo.utils.string_tools` into `gemseo.util.string`, and the call with `None` raises `TypeError: 'NoneType' object is not iterable`.

Pass an empty tuple instead of `None`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: 'NoneType' object is not iterable`
    - **See** merge request [!2795](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2795).

=== "GEMSEO 6.3"

    ```python
    from gemseo.utils.string_tools import filter_names
    print(list(filter_names(["a", "b", "c"], None)))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.util.string import filter_names
    print(list(filter_names(["a", "b", "c"], ())))
    ```

### The `GEMSEO_FAST` environment variable was removed

The `fast` option of the global configuration and the `GEMSEO_FAST` environment variable were removed; use the methods `enable_fast_mode` and `disable_fast_mode` of the global configuration instead, which disable the options costing time at every evaluation and reset them.
bump-gemseo rewrites the assignments of `configuration.fast` in the code into these methods, but it cannot see the `GEMSEO_FAST` environment variable, set outside the code or in a dotenv file; with GEMSEO 7, GEMSEO fails at import with an error naming these methods.
Stop setting `GEMSEO_FAST` and call `configuration.enable_fast_mode()` in the code, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `CalledProcessError: Command '['/usr/local/bin/python', '-c', 'from gemseo import configuration; print(configuration.enable_discipline_cache, configuration.validate_input_data)']' returned non-zero exit status 1.`
    - **See** issue [#1914](https://gitlab.com/gemseo/dev/gemseo/-/issues/1914), merge request [!2746](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2746).

=== "GEMSEO 6.3"

    ```python
    import os
    import subprocess
    import sys
    code = "from gemseo import configuration; print(configuration.enable_discipline_cache, configuration.validate_input_data)"
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "GEMSEO_FAST": "True"}, capture_output=True, text=True, check=True)
    print(result.stdout.split())
    ```

=== "GEMSEO 7"

    ```python
    import subprocess
    import sys
    code = "from gemseo import configuration; configuration.enable_fast_mode(); print(configuration.enable_discipline_cache, configuration.validate_input_data)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    print(result.stdout.split())
    ```

### The `step` of a gradient approximator can no longer be `None`

The `step` argument of `BaseGradientApproximator` can no longer be `None`; use `0.0` instead to use the default step.
bump-gemseo leaves the argument without a comment and the approximator fails with `AttributeError: 'NoneType' object has no attribute 'real'`.
Replace `step=None` by `step=0.0`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'NoneType' object has no attribute 'real'`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2309](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2309).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.utils.derivatives.finite_differences import FirstOrderFD
    approximator = FirstOrderFD(lambda x: x**2, step=None)
    print([approximator.step, approximator.f_gradient(np.array([3.0]))])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.util.derivative.approximator.forward_differences import ForwardDifferences
    approximator = ForwardDifferences(lambda x: x**2, step=0.0)
    print([approximator.step, approximator.f_gradient(np.array([3.0]))])
    ```

### Settings models validate the fields assigned after their creation

In GEMSEO 7, the settings models (`BaseSettings` and its subclasses, for the algorithms, formulations, MDAs and post-processors) validate a field when it is assigned after the creation of the model, not only at creation time.
Assigning a value of a wrong type now raises a `ValidationError`.
A `model_validator(mode="after")` that assigns a field is also run again by this assignment, which assigns the field again, and so on.
bump-gemseo changes nothing in such a validator, and creating the settings raises `RecursionError: maximum recursion depth exceeded`.

In a validator, set the value without validation, with `self.__dict__["field"] = value`, as in the example below.
Outside a validator, assign only values of the type of the field.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `RecursionError: maximum recursion depth exceeded`
    - **See** merge request [!2511](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2511).

=== "GEMSEO 6.3"

    ```python
    from pydantic import model_validator
    from gemseo.settings.base_settings import BaseSettings

    class Tolerance_Settings(BaseSettings):
        tolerance: float = 1e-6

        @model_validator(mode="after")
        def _clip(self):
            self.tolerance = max(self.tolerance, 1e-12)
            return self

    print(Tolerance_Settings(tolerance=0.0).tolerance)
    ```

=== "GEMSEO 7"

    ```python
    from pydantic import model_validator
    from gemseo.util.pydantic import BaseSettings

    class Tolerance_Settings(BaseSettings):
        tolerance: float = 1e-6

        @model_validator(mode="after")
        def _clip(self):
            self.__dict__["tolerance"] = max(self.tolerance, 1e-12)
            return self

    print(Tolerance_Settings(tolerance=0.0).tolerance)
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### The logging settings moved to the global configuration

`gemseo.utils.logging.LOGGING_SETTINGS` and `gemseo.utils.logging.LoggingSettings` were removed: use `gemseo.configuration.logging` instead of the former, and `gemseo.util.logging.LoggingConfiguration`, whose instance is `gemseo.configuration.logging`, instead of the latter.
`gemseo.algos.parameter_space.LOGGER` was removed too; use `logging.getLogger("gemseo.space.random")` instead.

bump-gemseo marks each import with a comment giving its replacement; left as is, the import raises an `ImportError` giving the same replacement.
Replace each name as the comment says, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `gemseo.algos.parameter_space.LOGGER: use logging.getLogger("gemseo.space.random") instead` / `gemseo.utils.logging.LOGGING_SETTINGS: use gemseo.configuration.logging instead` / `gemseo.utils.logging.LoggingSettings: use gemseo.util.logging.LoggingConfiguration, whose instance is gemseo.configuration.logging, instead`
    - **Error with GEMSEO 7**: `ImportError: The attribute 'LOGGER' of the module 'gemseo.algos.parameter_space' was removed; use logging.getLogger("gemseo.space.random") instead.`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2303](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2303).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.parameter_space import LOGGER
    from gemseo.utils.logging import LOGGING_SETTINGS
    from gemseo.utils.logging import LoggingSettings
    print([LOGGING_SETTINGS.date_format, LoggingSettings().message_format == LOGGING_SETTINGS.message_format, LOGGER.name.split(".")[0]])
    ```

=== "GEMSEO 7"

    ```python
    import logging
    from gemseo import configuration
    from gemseo.util.logging import LoggingConfiguration
    LOGGER = logging.getLogger("gemseo.space.random")
    print([configuration.logging.date_format, LoggingConfiguration().message_format == configuration.logging.message_format, LOGGER.name.split(".")[0]])
    ```

### `DesignSpace.DesignVariableType` was replaced by `gemseo.enum.VariableType`

The `strenum` dependency was removed and the GEMSEO enumerations derive from `enum.StrEnum`, with the same member names and values; `enum.StrEnum` is not a subclass of `strenum.StrEnum`, so code annotated with or testing against `strenum.StrEnum` must use `enum.StrEnum`, which bump-gemseo does.
In addition, the class attribute `DesignSpace.DesignVariableType` was removed; use `gemseo.enum.VariableType` instead.

bump-gemseo marks the uses of `DesignSpace.DesignVariableType` with the comment "use gemseo.enum.VariableType instead"; left as is, they raise `AttributeError: type object 'DesignSpace' has no attribute 'DesignVariableType'`.
Use `VariableType`, whose member `REAL` replaces `FLOAT`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `use gemseo.enum.VariableType instead`
    - **Error with GEMSEO 7**: `AttributeError: type object 'DesignSpace' has no attribute 'DesignVariableType'`
    - **See** merge request [!2768](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2768).

=== "GEMSEO 6.3"

    ```python
    from strenum import StrEnum
    from gemseo.algos.design_space import DesignSpace
    print([isinstance(DesignSpace.DesignVariableType.FLOAT, StrEnum), issubclass(DesignSpace.DesignVariableType, StrEnum)])
    ```

=== "GEMSEO 7"

    ```python
    from enum import StrEnum
    from gemseo.enum import VariableType
    print([isinstance(VariableType.REAL, StrEnum), issubclass(VariableType, StrEnum)])
    ```

### `image_comparison` is replaced by `snapshot_matplotlib`

The `image_comparison` function of `gemseo.utils.testing.helpers` and the `baseline_images` fixture of `gemseo.utils.testing.pytest_conftest` were removed; the tests of plugins can use the `snapshot_matplotlib` fixture of syrupy-matplotlib instead.
bump-gemseo marks the imports with comments giving the steps; left as is, the import raises an `ImportError` giving them too.

Add syrupy-matplotlib to the test dependencies, set `snapshot_matplotlib_tolerance = 0.01` in `.pytest.ini`, replace the `@image_comparison([...])` decorator by a `snapshot_matplotlib` argument of the test, then run `pytest --snapshot-update` without `-n`; the `baseline_images` directories are no longer used.
If you import `gemseo.util.testing.pytest_conftest`, also add syrupy to the test dependencies, as this module imports it.

!!! info "After bump-gemseo"

    - **Comment**: `gemseo.utils.testing.helpers.image_comparison: use the snapshot_matplotlib fixture of syrupy-matplotlib instead; add syrupy-matplotlib to the test dependencies, set snapshot_matplotlib_tolerance = 0.01 in .pytest.ini, replace the @image_comparison([...]) decorator by a snapshot_matplotlib argument of the test, then run pytest --snapshot-update without -n (the baseline_images directories are no longer used)` / `gemseo.utils.testing.pytest_conftest.baseline_images: the baseline_images fixture went with image_comparison; use the snapshot_matplotlib fixture of syrupy-matplotlib instead` / `gemseo.utils.testing.pytest_conftest: add syrupy to the test dependencies of the project, as gemseo.util.testing.pytest_conftest imports it`
    - **Error with GEMSEO 7**: `ImportError: The attribute 'image_comparison' of the module 'gemseo.utils.testing.helpers' was removed; use the snapshot_matplotlib fixture of syrupy-matplotlib instead; add syrupy-matplotlib to the test dependencies ...`
    - **See** issue [#1760](https://gitlab.com/gemseo/dev/gemseo/-/issues/1760), merge request [!2479](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2479).

=== "GEMSEO 6.3"

    ```python
    from gemseo.utils.testing.helpers import image_comparison
    from gemseo.utils.testing.pytest_conftest import baseline_images
    print([callable(image_comparison), callable(baseline_images)])
    ```

=== "GEMSEO 7"

    No GEMSEO 7 code is shown for this example: follow the explanation above.

## Other comments of bump-gemseo

bump-gemseo also writes these comments, on code this page does not illustrate.

- `GlobalConfiguration.__init__(fast=...)`: `call configuration.enable_fast_mode() after creating it`

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. These names have no direct replacement: unless an entry above or the changelog gives one, remove the code that uses them.

- `gemseo.utils.compatibility.matplotlib`
- `gemseo.utils.compatibility.openturns.OT_1_23`
- `gemseo.utils.compatibility.openturns.PEARSON_METHOD_NAME`
- `gemseo.utils.compatibility.openturns.create_trend_basis`
- `gemseo.utils.compatibility.scipy.SCIPY_GREATER_THAN_1_14`
- `gemseo.utils.compatibility.scipy.SCIPY_LOWER_THAN_1_12`
- `gemseo.utils.compatibility.scipy.TOL_OPTION`
- `gemseo.utils.discipline.DummyBaseDiscipline`
- `gemseo.utils.enumeration`
- `gemseo.utils.locks.synchronized_hashes`
- `gemseo.utils.study_analyses.study_analysis_cli.STUDY_ANALYSIS_TYPES`
- `gemseo.utils.testing.bilevel_test_helper`
- `gemseo.utils.testing.pytest_conftest.disable_fault_handler`
- `gemseo.utils.testing.pytest_conftest.import_or_skip_xlwings`
- `gemseo.utils.testing.pytest_conftest.is_xlwings_usable`
- `gemseo.utils.testing.pytest_conftest.original_image_directories`
- `gemseo.utils.testing.pytest_conftest.skip_if_xlwings_is_not_usable`
- `gemseo.utils.testing.pytest_conftest.skip_if_xlwings_is_usable`

??? note "Private names (3), used by some plugins"

    - `gemseo.utils.discipline._add_to_sub`
    - `gemseo.utils.testing.pytest_conftest.__ctx_tmp_wd`
    - `gemseo.utils.testing.pytest_conftest._image_directories`
