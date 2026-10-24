---
status: new
description: "Migrating scenarios from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Scenarios

The changes of GEMSEO 7 about scenarios, scenario adapters and their results that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### `BackupSettings` takes keyword arguments only

In GEMSEO 7, `BackupSettings` is a Pydantic model instead of a dataclass: it can no longer be built with positional arguments, and unknown setting names raise a validation error instead of being ignored.

bump-gemseo leaves the call unchanged and writes no comment, so it fails with `TypeError: BaseModel.__init__() takes 1 positional argument but 3 were given`.
Name each argument, e.g. `BackupSettings(file_path=...)`, and `at_each_iteration` for the second positional argument of the example below.
Check also the spelling of the names, since an unknown name, ignored by GEMSEO 6.3, is now an error.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: BaseModel.__init__() takes 1 positional argument but 3 were given`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2006](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2006).

=== "GEMSEO 6.3"

    ```python
    from gemseo.scenarios.backup_settings import BackupSettings
    settings = BackupSettings("backup.h5", True)
    print([str(settings.file_path), settings.at_each_iteration, settings.at_each_function_call])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.scenario.backup_settings import BackupSettings
    settings = BackupSettings(file_path="backup.h5", at_each_iteration=True)
    print([str(settings.file_path), settings.at_each_iteration, settings.at_each_function_call])
    ```

### `clear_history_before_run` was removed

In GEMSEO 6.3, a scenario had an attribute `clear_history_before_run`, set to `False` and read nowhere: setting it to `True` did not clear anything.
GEMSEO 7 removed it; the attribute that clears the database of the problem before each execution is `clear_database_before_execute`, formerly `clear_history_before_execute`.

bump-gemseo leaves the name unchanged and writes no comment: reading the attribute fails with `AttributeError: 'MDOScenario' object has no attribute 'clear_history_before_run'`, while setting it still does nothing.
Remove the code that uses it; set `clear_database_before_execute=True` only if you want the database cleared, which GEMSEO 6.3 did not do with this attribute.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'MDOScenario' object has no attribute 'clear_history_before_run'`
    - **See** issue [#1666](https://gitlab.com/gemseo/dev/gemseo/-/issues/1666), merge request [!2171](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2171).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.scenarios.mdo_scenario import MDOScenario
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = MDOScenario([AnalyticDiscipline({"y": "2*x"})], "y", space, formulation_name="DisciplinaryOpt")
    print(scenario.clear_history_before_run)
    ```

=== "GEMSEO 7"

    No GEMSEO 7 code is shown for this example: follow the explanation above.

### Sample a random space with an `EvaluationScenario`

In GEMSEO 6.3, a `DOEScenario` could sample a `ParameterSpace`, such as the `IshigamiSpace` of the example.
GEMSEO 7 replaced `ParameterSpace` by `RandomSpace`, and a `RandomSpace` is sampled with an `EvaluationScenario`, which evaluates observables instead of an objective.

bump-gemseo makes this change on its own: it turns the `DOEScenario` over a `ParameterSpace` into an `EvaluationScenario` and the objective into an observable, without comment.
What it leaves is the name of the group of the sampled inputs in the dataset of the scenario: `DESIGN_GROUP` does not exist for this dataset, and the code fails with `AttributeError: 'IODataset' object has no attribute 'DESIGN_GROUP'`.
Read the sampled inputs from the `input_group` of the dataset, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'IODataset' object has no attribute 'DESIGN_GROUP'`
    - **See** issue [#1666](https://gitlab.com/gemseo/dev/gemseo/-/issues/1666), merge request [!2171](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2171); merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
    from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
    from gemseo.scenarios.doe_scenario import DOEScenario
    scenario = DOEScenario([IshigamiDiscipline()], "y", IshigamiSpace(), formulation_name="DisciplinaryOpt")
    scenario.execute(algo_name="OT_MONTE_CARLO", n_samples=20, seed=4)
    dataset = scenario.to_dataset()
    print([len(dataset), sorted(dataset.get_variable_names(dataset.DESIGN_GROUP))])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import OT_MONTE_CARLO_Settings
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
    from gemseo.problem.uncertainty.ishigami import IshigamiSpace
    from gemseo.scenario import EvaluationScenario
    scenario = EvaluationScenario([IshigamiDiscipline()], IshigamiSpace(), formulation_settings=DisciplinaryOpt_Settings())
    scenario.add_observable("y")
    scenario.execute(OT_MONTE_CARLO_Settings(n_samples=20, seed=4))
    dataset = scenario.to_dataset()
    print([len(dataset), sorted(dataset.get_variable_names(dataset.input_group))])
    ```

### `ScenarioResult.plot` takes a settings model

In GEMSEO 6.3, `ScenarioResult.plot` took the name of a post-processor and its settings as keyword arguments.
In GEMSEO 7, it takes a single argument, the settings model of the post-processor.

bump-gemseo leaves the call unchanged and writes no comment, so it fails with `TypeError: ScenarioResult.plot() got an unexpected keyword argument 'variable_names'`.
Replace the name and the keyword arguments by the settings model of the post-processor built from these arguments, e.g. `BasicHistory_Settings` for `"BasicHistory"`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: ScenarioResult.plot() got an unexpected keyword argument 'variable_names'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.scenarios.mdo_scenario import MDOScenario
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=1.0, value=0.9)
    scenario = MDOScenario([AnalyticDiscipline({"y": "(a - 0.25)**2"})], "y", space, formulation_name="DisciplinaryOpt")
    scenario.execute(algo_name="SLSQP", max_iter=10)
    post = scenario.get_result().plot("BasicHistory", variable_names=["y"], save=False, show=False)
    print(len(post.figures))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.post import BasicHistory_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=1.0, value=0.9)
    scenario = MDOScenario([AnalyticDiscipline({"y": "(a - 0.25)**2"})], space, formulation_settings=DisciplinaryOpt_Settings())
    scenario.add_objective("y")
    scenario.execute(SLSQP_Settings(max_iter=10))
    post = scenario.get_result().plot(BasicHistory_Settings(variable_names=["y"], save=False, show=False))
    print(len(post.figures))
    ```

### `MDOScenario.Settings` was removed

In GEMSEO 6.3, the class attribute `Settings` of a scenario, e.g. `MDOScenario.Settings`, built a model gathering `algo_name` and `algo_settings`.
GEMSEO 7 removed it: the single argument of `MDOScenario.execute` is `algorithm_settings`, the settings model of the algorithm.

bump-gemseo leaves it unchanged and writes no comment, so the code fails with `AttributeError: type object 'MDOScenario' has no attribute 'Settings'`.
Remove these uses and pass the settings model of the algorithm to `execute` instead; as the example below shows, there is no attribute to read in its place.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: type object 'MDOScenario' has no attribute 'Settings'`
    - **See** issue [#1666](https://gitlab.com/gemseo/dev/gemseo/-/issues/1666), merge request [!2333](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2333).

=== "GEMSEO 6.3"

    ```python
    from gemseo.scenarios.mdo_scenario import MDOScenario
    settings = MDOScenario.Settings(algo_name="SLSQP", algo_settings={"max_iter": 3})
    print([settings.algo_name, dict(settings.algo_settings)])
    ```

=== "GEMSEO 7"

    No GEMSEO 7 code is shown for this example: follow the explanation above.

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### `MDOObjectiveScenarioAdapter` was replaced by `output_optimal_objective=True` { #scenario-adapters }

GEMSEO 7 removed `MDOObjectiveScenarioAdapter`; use `MDOScenarioAdapter` with `output_optimal_objective=True`, which reports the optimal objective value of the scenario instead of the one computed by the disciplines at the optimum.
The arguments of the scenario adapters also no longer refer to an optimization: e.g. `reset_x0_before_opt` and `keep_opt_history` are renamed `reset_x0_before_exec` and `keep_databases`, and constants such as `LOWER_BND_SUFFIX` become `lower_bnd_suffix`.

bump-gemseo renames these arguments and attributes, but cannot migrate the class and marks the code using it with a `TODO(bump-gemseo)` comment; left as it is, the import raises an `ImportError`.
Import `MDOScenarioAdapter` from `gemseo.scenario`, replace the class name and add `output_optimal_objective=True`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `gemseo.disciplines.scenario_adapters.mdo_objective_scenario_adapter.MDOObjectiveScenarioAdapter: use gemseo.scenario.adapter.mdo.MDOScenarioAdapter with output_optimal_objective=True instead`
    - **Error with GEMSEO 7**: `ImportError: The attribute 'MDOObjectiveScenarioAdapter' of the module 'gemseo.disciplines.scenario_adapters.mdo_objective_scenario_adapter' was removed; use gemseo.scenario.adapter.mdo.MDOScenarioAdapter with ...`
    - **See** issue [#1888](https://gitlab.com/gemseo/dev/gemseo/-/issues/1888), merge request [!2733](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2733).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.disciplines.scenario_adapters.mdo_objective_scenario_adapter import MDOObjectiveScenarioAdapter
    from gemseo.scenarios.mdo_scenario import MDOScenario
    discipline = AnalyticDiscipline({"y": "(a - b)**2 + 1"})
    space = DesignSpace()
    space.add_variable("a", lower_bound=-1.0, upper_bound=1.0, value=0.9)
    scenario = MDOScenario([discipline], "y", space, formulation_name="DisciplinaryOpt")
    scenario.set_algorithm(algo_name="SLSQP", max_iter=30)
    adapter = MDOObjectiveScenarioAdapter(scenario, ["b"], ["y"], reset_x0_before_opt=True, set_x0_before_opt=False, set_bounds_before_opt=False, keep_opt_history=True, save_opt_history=False, opt_history_file_prefix="adapter")
    output = adapter.execute({"b": np.array([0.4])})
    print([output["y"], adapter.keep_opt_history, adapter.save_opt_history, [MDOObjectiveScenarioAdapter.LOWER_BND_SUFFIX, MDOObjectiveScenarioAdapter.UPPER_BND_SUFFIX, MDOObjectiveScenarioAdapter.MULTIPLIER_SUFFIX, MDOObjectiveScenarioAdapter.DEFAULT_DATABASE_FILE_PREFIX]])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.scenario import MDOScenarioAdapter
    from gemseo.space import DesignSpace
    discipline = AnalyticDiscipline({"y": "(a - b)**2 + 1"})
    space = DesignSpace()
    space.add_variable("a", lower_bound=-1.0, upper_bound=1.0, value=0.9)
    scenario = MDOScenario([discipline], space, formulation_settings=DisciplinaryOpt_Settings())
    scenario.add_objective("y")
    scenario.set_algorithm(SLSQP_Settings(max_iter=30))
    adapter = MDOScenarioAdapter(scenario, ["b"], ["y"], reset_x0_before_exec=True, set_x0_before_exec=False, set_bounds_before_exec=False, keep_databases=True, save_databases=False, database_file_prefix="adapter", output_optimal_objective=True)
    output = adapter.execute({"b": np.array([0.4])})
    print([output["y"], adapter.keep_databases, adapter.save_databases, [MDOScenarioAdapter.lower_bnd_suffix, MDOScenarioAdapter.upper_bnd_suffix, MDOScenarioAdapter.multiplier_suffix, MDOScenarioAdapter.default_database_file_prefix]])
    ```

### The GGobi export uses `to_ggobi`

In GEMSEO 6.3, `save_optimization_history` wrote the history in the format set by `file_format`, e.g. `"ggobi"`.
In GEMSEO 7, it was split into `MDOScenario.to_hdf` and `MDOScenario.to_ggobi`, and `file_format` was removed.

bump-gemseo rewrites the call with `to_hdf` and marks the line with a comment recalling that `to_hdf(file_path)` writes the HDF5 format.
Left as it is, the file is an HDF5 file instead of a GGobi one, so the code reading it as text fails, here with `UnicodeDecodeError`.
If the file format was `"ggobi"`, replace `to_hdf` by `to_ggobi`, as in the example below, and remove the comment.

!!! info "After bump-gemseo"

    - **Comment**: `to_hdf(file_path) writes the HDF5 format; if the file format was ggobi, use to_ggobi(file_path) instead`
    - **Error with GEMSEO 7**: `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte`
    - **See** issue [#1666](https://gitlab.com/gemseo/dev/gemseo/-/issues/1666), merge request [!2333](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2333).

=== "GEMSEO 6.3"

    ```python
    from pathlib import Path
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.scenarios.mdo_scenario import MDOScenario
    discipline = AnalyticDiscipline({"y": "(a - 0.25)**2"})
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=1.0, value=0.9)
    scenario = MDOScenario([discipline], "y", space, formulation_name="DisciplinaryOpt")
    scenario.execute(algo_name="SLSQP", max_iter=10)
    scenario.save_optimization_history("history.xml", file_format="ggobi")
    print(Path("history.xml").read_text().count("<record>") > 0)
    ```

=== "GEMSEO 7"

    ```python
    from pathlib import Path
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.space import DesignSpace
    discipline = AnalyticDiscipline({"y": "(a - 0.25)**2"})
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=1.0, value=0.9)
    scenario = MDOScenario([discipline], space, formulation_settings=DisciplinaryOpt_Settings())
    scenario.add_objective("y")
    scenario.execute(SLSQP_Settings(max_iter=10))
    scenario.to_ggobi("history.xml")
    print(Path("history.xml").read_text().count("<record>") > 0)
    ```

### `BaseScenario._settings` was replaced by `_algorithm_settings`

In GEMSEO 6.3, the private attribute `_settings` of a scenario held the `algo_name` and the `algo_settings` passed to `execute`.
In GEMSEO 7, the single argument of `MDOScenario.execute` is `algorithm_settings`, a settings model, and `_settings` was replaced by `_algorithm_settings`: the settings model of the driver given to `execute`, or `None` before.

bump-gemseo marks the line with a comment; left as it is, the code raises `AttributeError: 'MDOScenario' object has no attribute '_settings'`.
Read the algorithm name from `_algorithm_settings.target_class_name` and each former item of `algo_settings` from the field of the same name, e.g. `_algorithm_settings.max_iter`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `_settings was replaced by _algorithm_settings, the settings model of the driver given to execute, or None before; read its target_class_name for algo_name and its fields for the items of algo_settings`
    - **Error with GEMSEO 7**: `AttributeError: 'MDOScenario' object has no attribute '_settings'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.scenarios.mdo_scenario import MDOScenario
    space = DesignSpace()
    space.add_variable("x", lower_bound=-1.0, upper_bound=1.0, value=0.5)
    scenario = MDOScenario([AnalyticDiscipline({"y": "x**2"})], "y", space, formulation_name="DisciplinaryOpt")
    scenario.execute(algo_name="SLSQP", max_iter=7)
    print([scenario._settings.algo_name, scenario._settings.algo_settings["max_iter"]])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", lower_bound=-1.0, upper_bound=1.0, value=0.5)
    scenario = MDOScenario([AnalyticDiscipline({"y": "x**2"})], space, formulation_settings=DisciplinaryOpt_Settings())
    scenario.add_objective("y")
    scenario.execute(SLSQP_Settings(max_iter=7))
    print([scenario._algorithm_settings.target_class_name, scenario._algorithm_settings.max_iter])
    ```

## Other comments of bump-gemseo

bump-gemseo also writes these comments, on code this page does not illustrate.

- `DOEScenario.__init__(formulation_settings=...)`: `pass the formulation settings as formulation_settings=<Formulation>_Settings(...), as the scenario takes no loose keyword`
- `MDOScenario.__init__(formulation_settings=...)`: `pass the formulation settings as formulation_settings=<Formulation>_Settings(...), as the scenario takes no loose keyword`
- `OptAsMDOScenario.__init__(formulation_settings=...)`: `pass the formulation settings as formulation_settings=<Formulation>_Settings(...), as the scenario takes no loose keyword`

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. These names have no direct replacement: unless an entry above or the changelog gives one, remove the code that uses them.

- `gemseo.scenarios.base_scenario.ScenarioInputDataType`
