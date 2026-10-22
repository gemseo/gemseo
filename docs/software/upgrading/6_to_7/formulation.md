---
description: "Migrating formulations from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Formulations

The changes of GEMSEO 7 about MDO formulations and custom formulations that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### `BiLevel` names the database files of a sub-scenario after its index

In GEMSEO 6.3, with the setting `save_opt_history=True`, the `BiLevel` formulation saved the database of each execution of a sub-scenario to an HDF5 file named after the sub-scenario and the execution number, e.g. `"Sub_1.h5"`.
GEMSEO 7 inserts the index of the sub-scenario between them, e.g. `"Sub_0_1.h5"` for the first execution of the first sub-scenario, so that two sub-scenarios with the same name no longer overwrite each other's files.

bump-gemseo writes no comment, and the code that reads these files by name fails with a `FileNotFoundError`; the code that searches them with a pattern, e.g. `glob("Sub_*.h5")`, finds the new names without error.
Insert the index of the sub-scenario, i.e. its position among the sub-scenarios of the formulation, in the file names, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `FileNotFoundError: Sub_1.h5 not found.`
    - **See** issue [#1968](https://gitlab.com/gemseo/dev/gemseo/-/issues/1968), merge request [!2786](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2786).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.database import Database
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.scenarios.mdo_scenario import MDOScenario
    discipline = AnalyticDiscipline({"f": "(x-1)**2+(y-2)**2"}, name="D")
    sub_space = DesignSpace()
    sub_space.add_variable("y", lower_bound=0.0, upper_bound=5.0, value=0.0)
    sub = MDOScenario([discipline], "f", sub_space, name="Sub", formulation_name="DisciplinaryOpt")
    sub.set_algorithm(algo_name="SLSQP", max_iter=5)
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=5.0, value=0.0)
    scenario = MDOScenario([sub], "f", space, formulation_name="BiLevel", save_opt_history=True)
    scenario.execute(algo_name="NLOPT_COBYLA", max_iter=2)
    print(len(Database.from_hdf("Sub_1.h5")))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.core.problem.database import Database
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.formulation import BiLevel_Settings
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.optimization import NLOPT_COBYLA_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.space import DesignSpace
    discipline = AnalyticDiscipline({"f": "(x-1)**2+(y-2)**2"}, name="D")
    sub_space = DesignSpace()
    sub_space.add_variable("y", lower_bound=0.0, upper_bound=5.0, value=0.0)
    sub = MDOScenario([discipline], sub_space, name="Sub", formulation_settings=DisciplinaryOpt_Settings())
    sub.add_objective("f")
    sub.set_algorithm(SLSQP_Settings(max_iter=5))
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=5.0, value=0.0)
    scenario = MDOScenario([sub], space, formulation_settings=BiLevel_Settings(save_opt_history=True))
    scenario.add_objective("f")
    scenario.execute(NLOPT_COBYLA_Settings(max_iter=2))
    print(len(Database.from_hdf("Sub_0_1.h5")))
    ```

### `get_sub_optimization_result` raises an error for a negative index

In GEMSEO 6.3, `BiLevelScenarioResult.get_sub_optimization_result` returned `None` for a negative index, as for a sub-scenario without optimization history.
GEMSEO 7 raises a `ValueError` instead, as it already did for an index greater than the number of sub-scenarios minus one; a negative index does not count from the end as in a Python list.

bump-gemseo writes no comment, and the code that relied on `None` fails with `ValueError: The index (-1) of a sub-scenario must be between 0 and 0.`
Check the index before the call, as in the example below, or pass the index from the start, e.g. `n - 1` instead of `-1` for the last of `n` sub-scenarios.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ValueError: The index (-1) of a sub-scenario must be between 0 and 0.`
    - **See** issue [#1968](https://gitlab.com/gemseo/dev/gemseo/-/issues/1968), merge request [!2786](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2786).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.scenarios.mdo_scenario import MDOScenario
    discipline = AnalyticDiscipline({"f": "(x-1)**2+(y-2)**2"}, name="D")
    sub_space = DesignSpace()
    sub_space.add_variable("y", lower_bound=0.0, upper_bound=5.0, value=0.0)
    sub = MDOScenario([discipline], "f", sub_space, name="Sub", formulation_name="DisciplinaryOpt")
    sub.set_algorithm(algo_name="SLSQP", max_iter=5)
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=5.0, value=0.0)
    scenario = MDOScenario([sub], "f", space, formulation_name="BiLevel", keep_opt_history=True)
    scenario.execute(algo_name="NLOPT_COBYLA", max_iter=2)
    result = scenario.get_result()
    print([result.get_sub_optimization_result(index) is None for index in (-1, 0)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.formulation import BiLevel_Settings
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.optimization import NLOPT_COBYLA_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.space import DesignSpace
    discipline = AnalyticDiscipline({"f": "(x-1)**2+(y-2)**2"}, name="D")
    sub_space = DesignSpace()
    sub_space.add_variable("y", lower_bound=0.0, upper_bound=5.0, value=0.0)
    sub = MDOScenario([discipline], sub_space, name="Sub", formulation_settings=DisciplinaryOpt_Settings())
    sub.add_objective("f")
    sub.set_algorithm(SLSQP_Settings(max_iter=5))
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=5.0, value=0.0)
    scenario = MDOScenario([sub], space, formulation_settings=BiLevel_Settings(keep_opt_history=True))
    scenario.add_objective("f")
    scenario.execute(NLOPT_COBYLA_Settings(max_iter=2))
    result = scenario.get_result()
    print([index < 0 or result.get_sub_optimization_result(index) is None for index in (-1, 0)])
    ```

### `BiLevel` level constants and formulation `ConstraintType` were removed

GEMSEO 7 removed the `BiLevel` attributes `SYSTEM_LEVEL`, `SUBSCENARIOS_LEVEL` and `LEVELS` without replacement: `BiLevel.create_constraint` uses the boolean arguments `apply_to_system_level` and `apply_to_sub_level` instead of the argument `levels`, which was a list of these literals, and you can pass them as keyword arguments to `MDOScenario.add_constraint`.
The enumeration `ConstraintType` of a formulation, e.g. `MDF.ConstraintType`, was removed too; use `MDOScenario.ConstraintType` or `ArrayFunction.ConstraintType`.

bump-gemseo leaves these names unchanged and writes no comment, so the code fails with `AttributeError: type object 'BiLevel' has no attribute 'SYSTEM_LEVEL'`.
Replace each level constant by the boolean argument it stood for, and take the constraint type from the scenario, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: type object 'BiLevel' has no attribute 'SYSTEM_LEVEL'`
    - **GEMSEO 7 cannot give the GEMSEO 6.3 value here**: the BiLevel level constants were removed without replacement (issue [#94](https://gitlab.com/gemseo/dev/gemseo/-/issues/94)).
    - **See** issue [#94](https://gitlab.com/gemseo/dev/gemseo/-/issues/94), merge request [!2171](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2171).

=== "GEMSEO 6.3"

    ```python
    from gemseo.formulations.bilevel import BiLevel
    from gemseo.formulations.mdf import MDF
    print([[BiLevel.SYSTEM_LEVEL, BiLevel.SUBSCENARIOS_LEVEL, list(BiLevel.LEVELS)], MDF.ConstraintType.INEQ])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.scenario import MDOScenario
    print(["<removed in GEMSEO 7>", MDOScenario.ConstraintType.INEQ])
    ```

### `IDF` requires finite bounds for the coupling targets to normalize the constraints

The setting `normalize_constraints` of `IDF`, `True` by default, divides each consistency constraint by the bound range of its target coupling variable.
In GEMSEO 6.3, a target without bounds had an infinite range, so its consistency constraint was always zero and the optimizer ignored it without warning.
GEMSEO 7 raises a `ValueError` instead, since `IDF` also accepts spaces whose variables have no finite range, e.g. a `RandomSpace` sampled by an `EvaluationScenario`.

bump-gemseo writes no comment, and the code fails with `ValueError: IDF formulation: the variables 'y1' have a non-finite bound range, so the consistency constraints cannot be scaled; use the setting normalize_constraints=False.`
Either give finite bounds to the coupling targets in the design space, or set `normalize_constraints=False`, as in the example below; in both cases the consistency constraints are no longer zero, so the optimization results change.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ValueError: IDF formulation: the variables 'y1' have a non-finite bound range, so the consistency constraints cannot be scaled; use the setting normalize_constraints=False.`
    - **GEMSEO 7 cannot give the GEMSEO 6.3 value here**: GEMSEO 6.3 divided the consistency constraints by the infinite bound range of the coupling targets, so that they were always zero..
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.scenarios.mdo_scenario import MDOScenario
    disciplines = [AnalyticDiscipline({"y1": "x+0.2*y2"}, name="D1"), AnalyticDiscipline({"y2": "0.5*y1+1", "f": "x**2+y2"}, name="D2")]
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("y1", value=0.0)
    space.add_variable("y2", value=0.0)
    scenario = MDOScenario(disciplines, "f", space, formulation_name="IDF")
    problem = scenario.formulation.optimization_problem
    print([constraint.evaluate(space.get_current_value()) for constraint in problem.constraints])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.formulation import IDF_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.space import DesignSpace
    disciplines = [AnalyticDiscipline({"y1": "x+0.2*y2"}, name="D1"), AnalyticDiscipline({"y2": "0.5*y1+1", "f": "x**2+y2"}, name="D2")]
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("y1", value=0.0)
    space.add_variable("y2", value=0.0)
    scenario = MDOScenario(disciplines, space, formulation_settings=IDF_Settings(normalize_constraints=False))
    scenario.add_objective("f")
    problem = scenario.formulation.problem
    print([constraint.evaluate(space.get_current_value()) for constraint in problem.constraints])
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### Custom formulations implement `_create_multidisciplinary_process`

In GEMSEO 6.3, a custom formulation could override the hook `_init_before_design_space_and_objective` to run code at construction.
GEMSEO 7 removed it: implement the abstract method `_create_multidisciplinary_process` instead, otherwise the formulation is abstract and the factory drops it.
A formulation deriving directly from `BaseFormulation` or `BaseMDOFormulation` must also implement the abstract method `_update_input_space`.
Since a formulation no longer receives the objective at construction, build what needs the objective in `create_objective`.

bump-gemseo marks the method with a comment, as well as the constructor call (pass the problem first, e.g. `OptimizationProblem(design_space)`, and the loose settings in `settings`) and the overrides of `get_top_level_disciplines`, which now takes an argument `include_sub_formulations`, `False` by default.
Left as it is, the code fails with `ValueError: The Pydantic model must be a DisciplinaryOpt_Settings; got DesignSpace.`, since the design space now takes the place of the `settings` argument.
Rename the method, keep the call to `super()`, and build the formulation from an `OptimizationProblem` and the disciplines, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `_init_before_design_space_and_objective was removed; implement the abstract method _create_multidisciplinary_process instead, or the formulation is abstract and the factory drops it, and build in create_objective what needs the objective, which the formulation no longer receives at construction; see the formulation section of the GEMSEO 7 upgrading guide` / `pass the problem first, e.g. OptimizationProblem(design_space), then set its objective; the loose settings go into settings=<Formulation>_Settings(...)` / `get_top_level_disciplines now takes an argument include_sub_formulations, False by default; an override must take it too`
    - **Error with GEMSEO 7**: `ValueError: The Pydantic model must be a DisciplinaryOpt_Settings; got DesignSpace.`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.formulations.disciplinary_opt import DisciplinaryOpt


    class Custom(DisciplinaryOpt):
        """A formulation counting the disciplines when it is created."""

        def _init_before_design_space_and_objective(self):
            super()._init_before_design_space_and_objective()
            self.hook_calls = len(self.disciplines)


    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    formulation = Custom([AnalyticDiscipline({"y": "2*x"}, name="D")], "y", space)
    print([formulation.hook_calls, [discipline.name for discipline in formulation.get_top_level_disciplines()]])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.formulation.disciplinary_opt import DisciplinaryOpt
    from gemseo.optimization import OptimizationProblem
    from gemseo.space import DesignSpace


    class Custom(DisciplinaryOpt):
        """A formulation counting the disciplines when it is created."""

        def _create_multidisciplinary_process(self):
            super()._create_multidisciplinary_process()
            self.hook_calls = len(self.disciplines)


    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    formulation = Custom(OptimizationProblem(space), [AnalyticDiscipline({"y": "2*x"}, name="D")])
    print([formulation.hook_calls, [discipline.name for discipline in formulation.get_top_level_disciplines()]])
    ```

### `BaseMDOFormulation` takes the space type as a second parameter

In GEMSEO 6.3, a custom MDO formulation derived from `BaseMDOFormulation[SettingsT]`, the space of variables being pinned to `DesignSpace`.
In GEMSEO 7, the MDO formulations operate on any `BaseVariableSpace`, so that an `EvaluationScenario` can sample a `RandomSpace` through them; the space of variables is therefore a second type parameter: `BaseMDOFormulation[SettingsT, SpaceT]`.

bump-gemseo marks the class with a comment; left as it is, the class definition raises a `TypeError` about too few arguments.
Add the second type parameter: `DesignSpace` when the formulation reads bounds or a current value, as in the example below, or a type variable bound to `BaseVariableSpace` when it uses the base API only.
As the comment says, `DesignSpace` must then be imported at run time.

!!! info "After bump-gemseo"

    - **Comment**: `gemseo.formulations.base_mdo_formulation.BaseMDOFormulation: BaseMDOFormulation takes the type of the input space as a second type parameter, so a class deriving from BaseMDOFormulation[T] must derive from BaseMDOFormulation[T, DesignSpace], DesignSpace being imported at run time`
    - **Error with GEMSEO 7**: `TypeError: Too few arguments for <class 'gemseo.formulation.core.base_mdo.BaseMDOFormulation'>; actual 1, expected at least 2`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
    from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings


    class Custom(BaseMDOFormulation[DisciplinaryOpt_Settings]):
        """A custom formulation."""


    print([Custom.__name__, Custom.__mro__[1].__name__])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.formulation.core.base_mdo import BaseMDOFormulation
    from gemseo.space import DesignSpace


    class Custom(BaseMDOFormulation[DisciplinaryOpt_Settings, DesignSpace]):
        """A custom formulation."""


    print([Custom.__name__, Custom.__mro__[1].__name__])
    ```

### Custom formulations implement `_update_input_space`

In GEMSEO 6.3, `BaseFormulation._update_design_space` was a method doing nothing, that a formulation could override to filter its design space.
GEMSEO 7 renamed it `_update_input_space`, since the formulation now works on any space of variables, and made it abstract: a custom formulation that did not override it is abstract, and the factory drops it.
bump-gemseo renames an override of `_update_design_space`, and GEMSEO 7 still calls an override under the old name with a `DeprecationWarning`; only the formulations without override are affected.

bump-gemseo writes no comment about this method, and the instantiation fails with `TypeError: Can't instantiate abstract class Custom without an implementation for abstract methods '_create_multidisciplinary_process', '_update_input_space'`.
Implement `_update_input_space`, doing nothing to keep the space of variables as it is, as in the example below; the abstract method `_create_multidisciplinary_process` has its own entry.

!!! info "After bump-gemseo"

    - **Comment**: `gemseo.formulations.base_mdo_formulation.BaseMDOFormulation: BaseMDOFormulation takes the type of the input space as a second type parameter, so a class deriving from BaseMDOFormulation[T] must derive from BaseMDOFormulation[T, DesignSpace], DesignSpace being imported at run time` / `get_top_level_disciplines now takes an argument include_sub_formulations, False by default; an override must take it too` / `pass the problem first, e.g. OptimizationProblem(design_space), then set its objective; the loose settings go into settings=<Formulation>_Settings(...)`
    - **Error with GEMSEO 7**: `TypeError: Can't instantiate abstract class Custom without an implementation for abstract methods '_create_multidisciplinary_process', '_update_input_space'`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from numpy import array
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
    from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings

    class Custom(BaseMDOFormulation):
        """A formulation that leaves the design space as it is."""

        Settings = DisciplinaryOpt_Settings

        def get_top_level_disciplines(self, include_sub_formulations=False):
            return self.disciplines

    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("z", lower_bound=0.0, upper_bound=1.0, value=0.5)
    formulation = Custom([AnalyticDiscipline({"y": "2*x"}, name="D")], "y", space)
    problem = formulation.optimization_problem
    print([problem.design_space.variable_names, problem.objective.evaluate(array([0.5, 0.5]))])
    ```

=== "GEMSEO 7"

    ```python
    from numpy import array
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.formulation.core.base_mdo import BaseMDOFormulation
    from gemseo.optimization import OptimizationProblem
    from gemseo.space import DesignSpace

    class Custom(BaseMDOFormulation):
        """A formulation that leaves the design space as it is."""

        settings_class = DisciplinaryOpt_Settings

        def _create_multidisciplinary_process(self):
            pass

        def _update_input_space(self):
            pass

        def get_top_level_disciplines(self, include_sub_formulations=False):
            return self.disciplines

    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("z", lower_bound=0.0, upper_bound=1.0, value=0.5)
    problem = OptimizationProblem(space)
    formulation = Custom(problem, [AnalyticDiscipline({"y": "2*x"}, name="D")])
    problem.objective = formulation.create_objective(["y"])
    print([list(problem.design_space.variables), problem.objective.evaluate(array([0.5, 0.5]))])
    ```

### Constraints are attached by the scenario, not the formulation

In GEMSEO 6.3, `BaseFormulation.add_constraint` created a constraint and attached it to the optimization problem.
In GEMSEO 7, a formulation is no longer responsible for attaching the constraint function to the problem: the method `create_constraint` replaces `add_constraint` and only creates the constraint function from output names, while `MDOScenario.add_constraint` attaches it.

bump-gemseo marks the line with a comment; left as it is, the code raises `AttributeError: 'MDF' object has no attribute 'add_constraint'`.
Call `add_constraint` on the scenario instead of its formulation, with the same arguments, as in the example below; note also that the attribute `optimization_problem` of a formulation is renamed `problem`.
In a custom formulation, implement `create_constraint`, returning the constraint function or `None`, instead of `add_constraint`.

!!! info "After bump-gemseo"

    - **Comment**: `a formulation no longer adds a constraint; implement create_constraint(...), returning the constraint function or None, which MDOScenario.add_constraint attaches to the problem; see the formulation section of the GEMSEO 7 upgrading guide`
    - **Error with GEMSEO 7**: `AttributeError: 'MDF' object has no attribute 'add_constraint'`
    - **See** issue [#94](https://gitlab.com/gemseo/dev/gemseo/-/issues/94), merge request [!2171](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2171).

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
    from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
    from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
    from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
    from gemseo.scenarios.mdo_scenario import MDOScenario
    scenario = MDOScenario([Sellar1(), Sellar2(), SellarSystem()], "obj", SellarDesignSpace(), formulation_name="MDF")
    scenario.formulation.add_constraint("c_1", constraint_type="ineq")
    scenario.formulation.add_constraint("c_2", constraint_type="ineq", positive=True, value=1.0)
    print(sorted(c.name for c in scenario.formulation.optimization_problem.constraints))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.formulation import MDF_Settings
    from gemseo.problem.mdo.sellar import Sellar1
    from gemseo.problem.mdo.sellar import Sellar2
    from gemseo.problem.mdo.sellar import SellarDesignSpace
    from gemseo.problem.mdo.sellar import SellarSystem
    from gemseo.scenario import MDOScenario
    scenario = MDOScenario([Sellar1(), Sellar2(), SellarSystem()], SellarDesignSpace(), formulation_settings=MDF_Settings())
    scenario.add_objective("obj")
    scenario.add_constraint("c_1", constraint_type="ineq")
    scenario.add_constraint("c_2", constraint_type="ineq", positive=True, value=1.0)
    print(sorted(c.name for c in scenario.formulation.problem.constraints))
    ```

### `add_observable` of a formulation takes a list of names

In GEMSEO 6.3, the argument `output_names` of `BaseFormulation.add_observable` also accepted a single name as a `str`.
In GEMSEO 7, it no longer supports `str`: a single name would be iterated character by character.

bump-gemseo marks the line with a comment. Wrap the name in a list, e.g. `add_observable([output_name])`, as in the example below, and remove the comment.
Left as it is, the code raises no error with a one-character name like `"z"`, but a name of several characters would be split into its characters; check every marked call.

!!! info "After bump-gemseo"

    - **Comment**: `add_observable of a formulation takes an iterable of output names and no longer a single one, which it would iterate character by character; wrap it in a list, e.g. add_observable([output_name])`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.scenarios.doe_scenario import DOEScenario
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = DOEScenario([AnalyticDiscipline({"y": "a**2", "z": "3*a"})], "y", space, formulation_name="DisciplinaryOpt")
    scenario.formulation.add_observable("z")
    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=3)
    print(sorted(scenario.formulation.optimization_problem.database.get_function_history("z").ravel().tolist()))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.doe import PYDOE_FULLFACT_Settings
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = MDOScenario([AnalyticDiscipline({"y": "a**2", "z": "3*a"})], space, formulation_settings=DisciplinaryOpt_Settings())
    scenario.add_objective("y")
    scenario.formulation.add_observable(["z"])
    scenario.execute(PYDOE_FULLFACT_Settings(n_samples=3))
    print(sorted(scenario.formulation.problem.database.get_function_history("z").ravel().tolist()))
    ```

### A formulation is built from a problem

In GEMSEO 6.3, a formulation was built from the disciplines, the objective name, the design space, `minimize_objective` and loose settings.
In GEMSEO 7, the arguments of `BaseFormulation` are `problem` (an `EvaluationProblem`), `disciplines` and `settings`; the formulation creates the objective function with `create_objective` but no longer attaches it to the problem, which is done by the `MDOScenario`.
The attribute `optimization_problem` is renamed `problem`, since it can be an `EvaluationProblem` or an `OptimizationProblem`.

bump-gemseo renames `settings_model` to `settings` and marks the constructor call with a comment; left as it is, the code raises `TypeError: BaseFormulation.__init__() got an unexpected keyword argument 'minimize_objective'`.
Create an `OptimizationProblem` from the design space and pass it first, followed by the disciplines and `settings=MDF_Settings()`; then set `problem.objective` from `formulation.create_objective(["obj"])` and set `problem.minimize_objective`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `pass the problem first, e.g. OptimizationProblem(design_space), then set its objective; the loose settings go into settings=<Formulation>_Settings(...)`
    - **Error with GEMSEO 7**: `TypeError: BaseFormulation.__init__() got an unexpected keyword argument 'minimize_objective'`
    - **See** issue [#94](https://gitlab.com/gemseo/dev/gemseo/-/issues/94), merge request [!2171](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2171); issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2348](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2348).

=== "GEMSEO 6.3"

    ```python
    from gemseo.formulations.mdf import MDF
    from gemseo.formulations.mdf_settings import MDF_Settings
    from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
    from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
    from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
    from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    formulation = MDF(disciplines, "obj", SellarDesignSpace(), minimize_objective=False, settings_model=MDF_Settings())
    problem = formulation.optimization_problem
    print([problem.objective.name, problem.minimize_objective, sorted(problem.design_space.variable_names)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.formulation import MDF_Settings
    from gemseo.formulation.mdf import MDF
    from gemseo.optimization import OptimizationProblem
    from gemseo.problem.mdo.sellar import Sellar1
    from gemseo.problem.mdo.sellar import Sellar2
    from gemseo.problem.mdo.sellar import SellarDesignSpace
    from gemseo.problem.mdo.sellar import SellarSystem
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    formulation = MDF(OptimizationProblem(SellarDesignSpace()), disciplines, settings=MDF_Settings())
    problem = formulation.problem
    problem.objective = formulation.create_objective(["obj"])
    problem.minimize_objective = False
    print([problem.objective.name, problem.minimize_objective, sorted(problem.design_space.variables)])
    ```

### `MDF_Settings.main_mda_name` was removed

In GEMSEO 6.3, the main MDA of the `MDF` formulation could be selected by name with the `main_mda_name` field of `MDF_Settings`.
GEMSEO 7 removed this field; use `main_mda_settings` instead, which no longer accepts a `dict` but only an instance of a settings class associated with an MDA, e.g. `MDAJacobi_Settings`.

bump-gemseo marks the line with a comment saying that `MDF_Settings.main_mda_name` was removed; left as it is, the code raises `AttributeError: 'MDF_Settings' object has no attribute 'main_mda_name'`.
Replace the name of the MDA by an instance of its settings class given to `main_mda_settings`, then remove the comment; no GEMSEO 7 code is shown below, since the removed field itself has nothing left to read.

!!! info "After bump-gemseo"

    - **Comment**: `MDF_Settings.main_mda_name was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'MDF_Settings' object has no attribute 'main_mda_name'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.formulations.mdf_settings import MDF_Settings
    print(MDF_Settings(main_mda_name="MDAJacobi").main_mda_name)
    ```

=== "GEMSEO 7"

    No GEMSEO 7 code is shown for this example: follow the explanation above.

## Other comments of bump-gemseo

bump-gemseo also writes these comments, on code this page does not illustrate.

- `gemseo.formulations.base_formulation.BaseFormulation`: `BaseFormulation takes the type of the input space as a second type parameter, so a class deriving from BaseFormulation[T] must derive from BaseFormulation[T, DesignSpace], DesignSpace being imported at run time`

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. These names have no direct replacement: unless an entry above or the changelog gives one, remove the code that uses them.

- `gemseo.formulations.bilevel_bcd_settings.copy_field_opt`
