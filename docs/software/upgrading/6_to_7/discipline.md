---
description: "Migrating disciplines from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Disciplines

The changes of GEMSEO 7 about disciplines, chains and wrappers that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Silent changes

The code runs after bump-gemseo, without error or comment, and gives other results.

### `execute` returns the output data only

`BaseDiscipline.execute()` now returns the output data only, where it previously returned the merged dictionary of inputs and outputs.
bump-gemseo changes nothing here: the code runs without error after the migration, but the returned dictionary no longer contains the inputs, so code that iterates over it, counts or stores its items gets other results.

Check each use of the value returned by `execute`: read the inputs from `discipline.input_data[name]` instead, or, if you need the GEMSEO 6.3 dictionary, merge the default inputs, `discipline.input_data` and the outputs, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1269](https://gitlab.com/gemseo/dev/gemseo/-/issues/1269), merge request [!2500](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2500).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.disciplines.analytic import AnalyticDiscipline
    discipline = AnalyticDiscipline({"y": "2*x + z"})
    data = discipline.execute({"x": np.array([1.0]), "z": np.array([0.5])})
    print([sorted(data), {name: data[name] for name in sorted(data)}])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.discipline import AnalyticDiscipline
    discipline = AnalyticDiscipline({"y": "2*x + z"})
    output = discipline.execute({"x": np.array([1.0]), "z": np.array([0.5])})
    data = {**discipline.io.input_grammar.defaults, **discipline.input_data, **output}
    print([sorted(data), {name: data[name] for name in sorted(data)}])
    ```

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### `AutoPyDiscipline` uses simple grammars built from the type hints

The default grammar type of the `AutoPyDiscipline` is now `SimpleGrammar`, built from the type hints of the function.
In GEMSEO 6.3, its grammars were JSON grammars, whose number type accepts an `int` for a `float`-annotated argument; a `SimpleGrammar` rejects it.

bump-gemseo leaves the code without a comment and the execution raises an `InvalidDataError`, e.g. `Bad type for x: <class 'int'> instead of <class 'float'>.`
Pass values of the annotated types, e.g. `3.0` instead of `3` for a `float`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `InvalidDataError: Grammar compute_discipline_input: validation failed. Bad type for x: <class 'int'> instead of <class 'float'>. Bad type for y: <class 'int'> instead of <class 'float'>.`
    - **See** issue [#1496](https://gitlab.com/gemseo/dev/gemseo/-/issues/1496), merge request [!2296](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2296).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.disciplines.auto_py import AutoPyDiscipline


    def compute(x: float = 1.0, y: float = 2.0) -> float:
        z = x * y
        return z


    discipline = AutoPyDiscipline(compute)
    output = discipline.execute({"x": 3, "y": 2})
    print(np.atleast_1d(output["z"]))
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.discipline import AutoPyDiscipline


    def compute(x: float = 1.0, y: float = 2.0) -> float:
        z = x * y
        return z


    discipline = AutoPyDiscipline(compute)
    output = discipline.execute({"x": 3.0, "y": 2.0})
    print(np.atleast_1d(output["z"]))
    ```

### `MDOChain.copy_jacs` and `MDOChain.reverse_chain_rule` were removed

The methods `copy_jacs` and `reverse_chain_rule` of `MDOChain`, renamed `DisciplineChain`, were removed with no public replacement.
`copy_jacs` returned a deep copy of a Jacobian dictionary; `reverse_chain_rule` added a discipline to the Jacobian of the chain in reverse mode, which the chain does itself when it is linearized.

bump-gemseo renames the class but leaves these methods without a comment, so using them raises `AttributeError: type object 'DisciplineChain' has no attribute 'copy_jacs'`.
Remove the code calling them; to copy a Jacobian, copy each of its arrays.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: type object 'DisciplineChain' has no attribute 'copy_jacs'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.core.chains.chain import MDOChain
    print([callable(MDOChain.copy_jacs), callable(MDOChain.reverse_chain_rule)])
    ```

=== "GEMSEO 7"

    No GEMSEO 7 code is shown for this example: follow the explanation above.

### `convert_array_to_data` takes `name_to_slice`

The `names_to_slices` argument of `BaseDataConverter.convert_array_to_data` was renamed to `name_to_slice`, like the other arguments following the `plural_to_plural` naming convention, renamed to follow the `key_to_value` convention.
bump-gemseo leaves the argument without a comment and the call raises `TypeError: BaseDataConverter.convert_array_to_data() got an unexpected keyword argument 'names_to_slices'`.
Rename the argument, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: BaseDataConverter.convert_array_to_data() got an unexpected keyword argument 'names_to_slices'`
    - **See** issue [#1797](https://gitlab.com/gemseo/dev/gemseo/-/issues/1797), merge request [!2486](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2486).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
    split = split_array_to_dict_of_arrays(np.array([1.0, 2.0, 3.0]), {"a": 2, "b": 1}, ["a", "b"])
    converter = AnalyticDiscipline({"y": "a + b"}).io.input_grammar.data_converter
    data = converter.convert_array_to_data(np.array([1.0, 2.0, 3.0]), names_to_slices={"a": slice(0, 2), "b": slice(2, 3)})
    print([split, data])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.util.data_conversion import split_array_to_dict_of_arrays
    split = split_array_to_dict_of_arrays(np.array([1.0, 2.0, 3.0]), {"a": 2, "b": 1}, ["a", "b"])
    converter = AnalyticDiscipline({"y": "a + b"}).io.input_grammar.data_converter
    data = converter.convert_array_to_data(np.array([1.0, 2.0, 3.0]), name_to_slice={"a": slice(0, 2), "b": slice(2, 3)})
    print([split, data])
    ```

### The naming arguments of `DiscFromExe` and name generators are `naming`

The `directory_naming_method` argument of `DiscFromExe` was renamed to `naming`, and so was the `naming_method` argument of `BaseNameGenerator` and `NameGenerator`; the `DirectoryNamingMethod` constant of `gemseo.utils.directory_creator` was renamed to `Naming` and the module to `gemseo.util.directory_creator`.
bump-gemseo leaves these arguments without a comment and the call raises `TypeError: DiscFromExe.__init__() got an unexpected keyword argument 'directory_naming_method'`.
Rename the arguments to `naming`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: DiscFromExe.__init__() got an unexpected keyword argument 'directory_naming_method'`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2355](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2355).

=== "GEMSEO 6.3"

    ```python
    import sys
    from pathlib import Path
    import numpy as np
    from gemseo.disciplines.wrappers.disc_from_exe import DiscFromExe
    from gemseo.utils.directory_creator import DirectoryNamingMethod
    from gemseo.utils.name_generator import NameGenerator
    here = Path.cwd()
    discipline = DiscFromExe(
        here / "exe_input.tmpl", here / "exe_output.tmpl", here / "exe_runs", f"{sys.executable} {here / 'exe_model.py'}",
        "exe_input.txt", "exe_output.txt", directory_naming_method=DirectoryNamingMethod.NUMBERED,
    )
    output = discipline.execute({"x": np.array([2.0])})
    generator = NameGenerator(naming_method=NameGenerator.Naming.NUMBERED)
    print([output["y"], sorted(p.name for p in (here / "exe_runs").iterdir()), generator.generate_name()])
    ```

=== "GEMSEO 7"

    ```python
    import sys
    from pathlib import Path
    import numpy as np
    from gemseo.discipline import DiscFromExe
    from gemseo.util.directory_creator import Naming
    from gemseo.util.name_generator import NameGenerator
    here = Path.cwd()
    discipline = DiscFromExe(
        here / "exe_input.tmpl", here / "exe_output.tmpl", here / "exe_runs", f"{sys.executable} {here / 'exe_model.py'}",
        "exe_input.txt", "exe_output.txt", naming=Naming.NUMBERED,
    )
    output = discipline.execute({"x": np.array([2.0])})
    generator = NameGenerator(naming=NameGenerator.Naming.NUMBERED)
    print([output["y"], sorted(p.name for p in (here / "exe_runs").iterdir()), generator.generate_name()])
    ```

### `JobSchedulerDisciplineWrapper` is now `JobSchedulerDiscipline`

`JobSchedulerDisciplineWrapper` was renamed to `JobSchedulerDiscipline` and `JobSchedulerDisciplineWrapperFactory` to `JobSchedulerDisciplineFactory`.
bump-gemseo does not change the class name passed as a string to the factory and leaves no comment, so the call raises `ImportError: The class JobSchedulerDisciplineWrapper is not available; the available ones are: JobSchedulerDiscipline, LSF and SLURM.`
Use the string `"JobSchedulerDiscipline"`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ImportError: The class JobSchedulerDisciplineWrapper is not available; the available ones are: JobSchedulerDiscipline, LSF and SLURM.`
    - **See** issue [#1687](https://gitlab.com/gemseo/dev/gemseo/-/issues/1687), merge request [!2561](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2561).

=== "GEMSEO 6.3"

    ```python
    from gemseo.disciplines.wrappers.job_schedulers.factory import JobSchedulerDisciplineWrapperFactory
    factory = JobSchedulerDisciplineWrapperFactory()
    print(["SLURM" in factory.class_names, factory.get_class("JobSchedulerDisciplineWrapper").__name__.replace("Wrapper", "")])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline.wrapper.job_scheduler.factory import JobSchedulerDisciplineFactory
    factory = JobSchedulerDisciplineFactory()
    print(["SLURM" in factory.class_names, factory.get_class("JobSchedulerDiscipline").__name__])
    ```

### The class names of the chains changed in the factories

The chain classes were renamed: `MDOChain` to `DisciplineChain`, `MDOParallelChain` to `ParallelDisciplineChain`, `MDOAdditiveChain` to `AdditiveDisciplineChain`, `MDOWarmStartedChain` to `WarmStartedDisciplineChain` and `MDOInitializationChain` to `InitializationDisciplineChain`.
bump-gemseo does not change the class names passed as strings to `create_discipline` or `DisciplineFactory`, and leaves no comment, so the call raises an `ImportError` starting with `ImportError: The class MDOChain is not available`.
Replace each old name by the new one in these strings, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ImportError: The class MDOChain is not available; the available ones are: AdditiveDisciplineChain, Aerodynamics, AnalyticDiscipline, ArrayBasedFunctionDiscipline, AutoPyDiscipline, Concatenater, ConstraintAggregation ...`
    - **See** issue [#1268](https://gitlab.com/gemseo/dev/gemseo/-/issues/1268), merge request [!2401](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2401).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo import create_discipline
    from gemseo.disciplines.analytic import AnalyticDiscipline
    first = AnalyticDiscipline({"y": "2*x"}, name="A")
    second = AnalyticDiscipline({"z": "y + 1"}, name="B")
    chain = create_discipline("MDOChain", disciplines=[first, second])
    parallel = create_discipline("MDOParallelChain", disciplines=[AnalyticDiscipline({"y": "2*x"}), AnalyticDiscipline({"w": "3*x"})], use_threading=True)
    additive = create_discipline("MDOAdditiveChain", disciplines=[AnalyticDiscipline({"y": "2*x"}), AnalyticDiscipline({"y": "3*x"})], outputs_to_sum=["y"])
    warm = create_discipline("MDOWarmStartedChain", disciplines=[AnalyticDiscipline({"y": "2*x"})], variable_names_to_warm_start=["y"])
    init = create_discipline("MDOInitializationChain", disciplines=[AnalyticDiscipline({"y": "2*x"}), AnalyticDiscipline({"z": "y+1"})])
    data = {"x": np.array([2.0])}
    print([chain.execute(data)["z"], parallel.execute(data)["w"], additive.execute(data)["y"], warm.execute(data)["y"], init.__class__.__name__.replace("MDOInitializationChain", "InitializationDisciplineChain")])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo import create_discipline
    from gemseo.discipline import AnalyticDiscipline
    first = AnalyticDiscipline({"y": "2*x"}, name="A")
    second = AnalyticDiscipline({"z": "y + 1"}, name="B")
    chain = create_discipline("DisciplineChain", disciplines=[first, second])
    parallel = create_discipline("ParallelDisciplineChain", disciplines=[AnalyticDiscipline({"y": "2*x"}), AnalyticDiscipline({"w": "3*x"})], use_threading=True)
    additive = create_discipline("AdditiveDisciplineChain", disciplines=[AnalyticDiscipline({"y": "2*x"}), AnalyticDiscipline({"y": "3*x"})], outputs_to_sum=["y"])
    warm = create_discipline("WarmStartedDisciplineChain", disciplines=[AnalyticDiscipline({"y": "2*x"})], variable_names_to_warm_start=["y"])
    init = create_discipline("InitializationDisciplineChain", disciplines=[AnalyticDiscipline({"y": "2*x"}), AnalyticDiscipline({"z": "y+1"})])
    data = {"x": np.array([2.0])}
    print([chain.execute(data)["z"], parallel.execute(data)["w"], additive.execute(data)["y"], warm.execute(data)["y"], init.__class__.__name__])
    ```

### `LinearCombination` no longer accepts `input_size=None`

The `input_size` argument of `LinearCombination` can no longer be `None`; use `1` instead.
bump-gemseo leaves the argument without a comment and the call raises `TypeError: Use () not None as shape arguments`.
Replace `input_size=None` by `input_size=1`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: Use () not None as shape arguments`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2309](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2309).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.disciplines.linear_combination import LinearCombination
    discipline = LinearCombination(["a", "b"], "c", input_coefficients={"a": 2.0}, input_size=None)
    output = discipline.execute({"a": np.array([1.0]), "b": np.array([3.0])})
    print(output["c"])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.discipline import LinearCombination
    discipline = LinearCombination(["a", "b"], "c", input_coefficients={"a": 2.0}, input_size=1)
    output = discipline.execute({"a": np.array([1.0]), "b": np.array([3.0])})
    print(output["c"])
    ```

### `XLSDiscipline` moved to the `gemseo-excel` plugin

`XLSDiscipline`, the discipline wrapping an Excel workbook, is no longer part of GEMSEO: it moved to the `gemseo-excel` plugin.
bump-gemseo rewrites the import `from gemseo.disciplines.wrappers.xls_discipline import XLSDiscipline` into `from gemseo_excel.xls_discipline import XLSDiscipline`, without a comment.
Without the plugin, this import raises `ModuleNotFoundError: No module named 'gemseo_excel'`, and `create_discipline("XLSDiscipline", ...)` raises `ImportError: The class XLSDiscipline is not available`.

Install the `gemseo-excel` plugin in the environment of GEMSEO 7; the rest of the code stays as bump-gemseo leaves it.

!!! info "References"

    - **See** issue [#1833](https://gitlab.com/gemseo/dev/gemseo/-/issues/1833).

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### `local_data` and `IO.data` are deprecated copies of the data

`Discipline.local_data` and `IO.data` are deprecated; use `Discipline.input_data` and `Discipline.output_data`, or `IO.input_data` and `IO.output_data`, instead.
The deprecated attributes still work, but return a fresh union of the input and output data and emit a `DeprecationWarning`, so setting, updating or removing an item through them has no effect, and an output that `_run` produces through them is missing.

bump-gemseo marks each use with a comment, e.g. "cannot transform: setting an item of discipline.local_data has no effect, discipline.local_data being a copy of the data; write it to discipline.io.output_data or discipline.io.input_data".
Write the outputs to `discipline.io.output_data` (in `_run`, return them or use `update_output_data`), write the inputs to `discipline.io.input_data`, and read the merged data with `discipline.io.get_merged_data()`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: setting an item of discipline.local_data has no effect, discipline.local_data being a copy of the data; write it to discipline.io.output_data or discipline.io.input_data` / `cannot transform: setting an item of discipline.io.data has no effect, discipline.io.data being a copy of the data; write it to discipline.io.output_data or discipline.io.input_data` / `local_data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or io.update_output_data(data)), write the inputs to input_data, and read input_data, output_data or io.get_merged_data()` / `IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data or get_merged_data()`
    - **With GEMSEO 7**: no error, but other results.

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.disciplines.analytic import AnalyticDiscipline
    discipline = AnalyticDiscipline({"y": "2*x"})
    discipline.execute({"x": np.array([3.0])})
    discipline.local_data["y"] = np.array([10.0])
    discipline.io.data["x"] = np.array([7.0])
    print([sorted(discipline.local_data), discipline.io.data["x"], discipline.io.get_output_data()["y"]])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.discipline.analytic import AnalyticDiscipline
    discipline = AnalyticDiscipline({"y": "2*x"})
    discipline.execute({"x": np.array([3.0])})
    discipline.io.output_data["y"] = np.array([10.0])
    discipline.io.input_data["x"] = np.array([7.0])
    print([sorted(discipline.io.get_merged_data()), discipline.io.input_data["x"], discipline.io.get_output_data()["y"]])
    ```

### Variable renaming distinguishes inputs and outputs

The `VariableRenamer` methods, the `VariableTranslation` class and the `rename_discipline_variables` function now differentiate inputs and outputs, so you must state for each translation whether it applies to an input or an output.
A translation is now `(discipline_name, is_input, variable_name, new_variable_name)`, for `from_translations`, `add_translation` and `VariableTranslation`; the CSV and spreadsheet files of `from_csv` and `from_spreadsheet` have an `is_input` column in second position, `True` or `False`; the values of `from_dictionary` and of the translators given to `rename_discipline_variables` are pairs `(input_name_to_new_input_name, output_name_to_new_output_name)`.
`add_translations_by_discipline` takes `input_name_to_new_input_name` and `output_name_to_new_output_name`, two positional arguments, instead of `variable_names_to_new_variable_names`, and `add_translations_by_variable` takes `discipline_name_and_io_mapping`, binding a `(variable_name, is_input)` pair to a discipline name, instead of `discipline_names_to_variable_names`.

bump-gemseo marks each call with a comment describing the new format; left as is, the code raises e.g. `TypeError: VariableTranslation.__new__() missing 1 required positional argument: 'new_variable_name'`.
Rewrite each call and each file accordingly, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `the tuples are now (discipline_name, is_input, variable_name, new_variable_name)` / `the values are now a pair (input_name_to_new_input_name, output_name_to_new_output_name)` / `the file now has an is_input column in second position, True or False` / `a translation is now (discipline_name, is_input, variable_name, new_variable_name)` / `split the mapping into input_name_to_new_input_name and output_name_to_new_output_name, two positional arguments` / `the values are now (variable_name, is_input) pairs; the argument is renamed discipline_name_and_io_mapping` / `the translators are now a mapping from the discipline name to a pair (input_name_to_new_input_name, output_name_to_new_output_name)`
    - **Error with GEMSEO 7**: `TypeError: VariableTranslation.__new__() missing 1 required positional argument: 'new_variable_name'`
    - **See** issue [#1659](https://gitlab.com/gemseo/dev/gemseo/-/issues/1659), merge request [!2609](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2609).

=== "GEMSEO 6.3"

    ```python
    import pandas as pd
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.utils.discipline import VariableRenamer
    from gemseo.utils.discipline import VariableTranslation
    from gemseo.utils.discipline import rename_discipline_variables
    pd.DataFrame([["A", "y", "y1"], ["B", "y", "y2"]]).to_csv("renaming.csv", header=False, index=False)
    pd.DataFrame([["A", "x", "x1"]]).to_excel("renaming.xlsx", header=False, index=False)
    first = VariableRenamer.from_translations(("A", "x", "x1"), VariableTranslation("B", "z", "z2"))
    second = VariableRenamer.from_dictionary({"A": {"x": "x1"}, "B": {"y": "y2"}})
    third = VariableRenamer.from_csv("renaming.csv")
    fourth = VariableRenamer.from_spreadsheet("renaming.xlsx")
    renamer = VariableRenamer()
    renamer.add_translation(VariableTranslation("A", "x", "x1"))
    renamer.add_translations_by_discipline("B", variable_names_to_new_variable_names={"z": "z2"})
    renamer.add_translations_by_variable("w", discipline_names_to_variable_names={"A": "y", "B": "y"})


    def names(translators):
        disciplines = [AnalyticDiscipline({"y": "2*x"}, name="A"), AnalyticDiscipline({"z": "y + 1"}, name="B")]
        rename_discipline_variables(disciplines, translators)
        return [[sorted(d.io.input_grammar), sorted(d.io.output_grammar)] for d in disciplines]


    print([names(r.translators) for r in (first, second, third, fourth, renamer)])
    ```

=== "GEMSEO 7"

    ```python
    import pandas as pd
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.util.discipline import VariableRenamer
    from gemseo.util.discipline import VariableTranslation
    from gemseo.util.discipline import rename_discipline_variables
    pd.DataFrame([["A", False, "y", "y1"], ["B", True, "y", "y2"]]).to_csv("renaming.csv", header=False, index=False)
    pd.DataFrame([["A", True, "x", "x1"]]).to_excel("renaming.xlsx", header=False, index=False)
    first = VariableRenamer.from_translations(("A", True, "x", "x1"), VariableTranslation("B", False, "z", "z2"))
    second = VariableRenamer.from_dictionary({"A": ({"x": "x1"}, {}), "B": ({"y": "y2"}, {})})
    third = VariableRenamer.from_csv("renaming.csv")
    fourth = VariableRenamer.from_spreadsheet("renaming.xlsx")
    renamer = VariableRenamer()
    renamer.add_translation(VariableTranslation("A", True, "x", "x1"))
    renamer.add_translations_by_discipline("B", {}, {"z": "z2"})
    renamer.add_translations_by_variable("w", discipline_name_and_io_mapping={"A": ("y", False), "B": ("y", True)})


    def names(translators):
        disciplines = [AnalyticDiscipline({"y": "2*x"}, name="A"), AnalyticDiscipline({"z": "y + 1"}, name="B")]
        rename_discipline_variables(disciplines, translators)
        return [[sorted(d.io.input_grammar), sorted(d.io.output_grammar)] for d in disciplines]


    print([names(r.translators) for r in (first, second, third, fourth, renamer)])
    ```

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. These names have no direct replacement: unless an entry above or the changelog gives one, remove the code that uses them.

- `gemseo.disciplines.wrappers.xls_discipline.cwd`

??? note "Private names (2), used by some plugins"

    - `gemseo.core.discipline.io._GRAMMAR_FACTORY`
    - `gemseo.disciplines.scenario_adapters.mdo_scenario_adapter._ProcessFlow`
