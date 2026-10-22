---
description: "Migrating problems and functions from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Problems and functions

The changes of GEMSEO 7 about evaluation and optimization problems, functions, databases, benchmark problems that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### Some constants of the example problems are now tuples

In GEMSEO 7, the constants of `gemseo.problem.mdo.scalable.linear.disciplines_generator` are renamed and change type: `LETTERS` is now `letters`, a `tuple` instead of a NumPy `ndarray`, and the `DESC_*` constants are now `desc_*`, tuples of tuples instead of lists of lists. Likewise, `NUMERICS` of `gemseo.discipline.wrapper.disc_from_exe` is now `numerics`, a `tuple` instead of a `list`.

bump-gemseo renames these constants without comment, but code relying on the old types fails, e.g. with `AttributeError: 'tuple' object has no attribute 'tolist'`. Convert them with `list()` or unpack them into a new list where your code needs a list, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'tuple' object has no attribute 'tolist'`
    - **See** merge request [!2698](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2698).

=== "GEMSEO 6.3"

    ```python
    from gemseo.disciplines.wrappers.disc_from_exe import NUMERICS
    from gemseo.problems.mdo.scalable.linear.disciplines_generator import DESC_3_DISC_WEAK
    from gemseo.problems.mdo.scalable.linear.disciplines_generator import LETTERS
    letters = LETTERS[:3].tolist()
    description = DESC_3_DISC_WEAK + [("D", ["x"], ["y"])]
    numerics = NUMERICS + ["e"]
    print([letters, len(description), "e" in numerics])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline.wrapper.disc_from_exe import numerics as NUMERIC_CHARACTERS
    from gemseo.problem.mdo.scalable.linear.disciplines_generator import desc_3_disc_weak
    from gemseo.problem.mdo.scalable.linear.disciplines_generator import letters as LETTER_NAMES
    letters = list(LETTER_NAMES[:3])
    description = [*desc_3_disc_weak, ("D", ["x"], ["y"])]
    numerics = [*NUMERIC_CHARACTERS, "e"]
    print([letters, len(description), "e" in numerics])
    ```

### `EvaluationProblem.to_dataset` no longer accepts ignored keyword arguments

In GEMSEO 6.3, `EvaluationProblem.to_dataset` accepted deprecated `**dataset_options` that it ignored, and `EvaluationProblem.get_functions` accepted `**kwargs`. GEMSEO 7 removes them, so an unknown keyword argument now raises a `TypeError`.

bump-gemseo leaves no comment, and a call such as the one of the example raises `TypeError: EvaluationProblem.to_dataset() got an unexpected keyword argument 'unused_option'`. Remove these keyword arguments, as in the example below; since they were ignored, the dataset is the same.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: EvaluationProblem.to_dataset() got an unexpected keyword argument 'unused_option'`
    - **See** merge request [!2171](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2171).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.evaluation_problem import EvaluationProblem
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    problem = EvaluationProblem(space)
    problem.database.store(np.array([0.5]), {"y": 1.0})
    dataset = problem.to_dataset(unused_option=True)
    print(len(dataset))
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.doe import EvaluationProblem
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    problem = EvaluationProblem(space)
    problem.database.store(np.array([0.5]), {"y": 1.0})
    dataset = problem.to_dataset()
    print(len(dataset))
    ```

### The space of an `EvaluationProblem` is read-only

In GEMSEO 6.3, you could replace the design space of an `EvaluationProblem` with `problem.design_space = ...`. In GEMSEO 7, this space is named `input_space`, since it is not a design space in general, and `EvaluationProblem.input_space` is read-only: the space is set at instantiation and can no longer be replaced afterwards, as the functions of the problem are defined over it.

bump-gemseo leaves no comment, and the assignment raises `AttributeError: property 'input_space' of 'EvaluationProblem' object has no setter`. Create a new problem from the new space instead, as in the example below, and add the functions to this new problem.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: property 'input_space' of 'EvaluationProblem' object has no setter`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.evaluation_problem import EvaluationProblem
    first = DesignSpace()
    first.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    second = DesignSpace()
    second.add_variable("y", size=3, lower_bound=0.0, upper_bound=1.0)
    problem = EvaluationProblem(first)
    problem.design_space = second
    print(problem.design_space.dimension)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import EvaluationProblem
    from gemseo.space import DesignSpace
    first = DesignSpace()
    first.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    second = DesignSpace()
    second.add_variable("y", size=3, lower_bound=0.0, upper_bound=1.0)
    problem = EvaluationProblem(first)
    problem = EvaluationProblem(second)
    print(problem.input_space.dimension)
    ```

### The parallel differentiation options of a problem are an attribute

In GEMSEO 6.3, the constructors of `EvaluationProblem` and `OptimizationProblem` accepted `**parallel_differentiation_options`, such as `n_processes`, and `OptimizationProblem` had an `is_linear` argument. GEMSEO 7 removes these arguments. The `is_linear` argument was useless, as the `OptimizationProblem.is_linear` property is deduced from the constraints and objective.

bump-gemseo removes `is_linear` but leaves the parallel differentiation options without comment, so the call raises `TypeError: OptimizationProblem.__init__() got an unexpected keyword argument 'n_processes'`. Remove these options from the call and assign them as a dictionary to the `parallel_differentiation_options` attribute of the problem, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: OptimizationProblem.__init__() got an unexpected keyword argument 'n_processes'`
    - **See** merge request [!2171](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2171).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.algos.design_space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=-1.0, upper_bound=1.0, value=0.5)
    problem = OptimizationProblem(space, is_linear=False, n_processes=2)
    problem.objective = MDOFunction(lambda x: np.array([x @ x]), "f")
    problem.database.store(np.array([0.5, 0.5]), {"f": 0.5})
    dataset = problem.to_dataset()
    print([problem.is_linear, problem.parallel_differentiation_options.get("n_processes"), len(dataset)])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.optimization import OptimizationProblem
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=-1.0, upper_bound=1.0, value=0.5)
    problem = OptimizationProblem(space)
    problem.parallel_differentiation_options = {"n_processes": 2}
    problem.objective = ArrayFunction(lambda x: np.array([x @ x]), "f")
    problem.database.store(np.array([0.5, 0.5]), {"f": 0.5})
    dataset = problem.to_dataset()
    print([problem.is_linear, problem.parallel_differentiation_options.get("n_processes"), len(dataset)])
    ```

### Read-only arrays in the Sobieski and springs problems

In GEMSEO 7, the design bounds of the Sobieski problem, `SobieskiBase.design_bounds` and `SobieskiProblem.design_bounds`, are read-only arrays. So are the initial state arrays `initial_time`, `initial_position` and `initial_velocity` of `gemseo.problem.ode.spring.springs_dynamics_discipline` (`INITIAL_TIME` and so on in GEMSEO 6.3), and so the default input data of `SpringsDynamicsDiscipline`.

bump-gemseo renames these constants but leaves no comment, and an in-place assignment raises `ValueError: assignment destination is read-only`. Copy the arrays to get mutable arrays before modifying them, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ValueError: assignment destination is read-only`
    - **See** merge request [!2698](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2698).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
    from gemseo.problems.ode.springs import springs_dynamics_discipline
    problem = SobieskiProblem()
    lower, upper = problem.design_bounds
    lower[0] = 0.05
    initial_time = springs_dynamics_discipline.INITIAL_TIME
    initial_time[0] = 0.0
    print([lower[0], float(initial_time[0])])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.problem.mdo.sobieski.standalone.problem import SobieskiProblem
    from gemseo.problem.ode.spring import springs_dynamics_discipline
    problem = SobieskiProblem()
    lower, upper = problem.design_bounds
    lower = lower.copy()
    lower[0] = 0.05
    initial_time = springs_dynamics_discipline.initial_time.copy()
    initial_time[0] = 0.0
    print([lower[0], float(initial_time[0])])
    ```

### New argument names of `RestrictedFunction`

In GEMSEO 7, the `restriction_indices` and `restriction_values` arguments of `RestrictedFunction` are renamed `frozen_indexes` and `frozen_values`, and a `name` argument is added; if empty, it defaults to `"{function.name}_restriction"`. The public attribute `restriction_values` is removed.

bump-gemseo leaves no comment, so a call with the old keyword arguments raises `TypeError: RestrictedFunction.__init__() got an unexpected keyword argument 'restriction_indices'`. Rename the keyword arguments and, if you read `restriction_values` afterwards, keep your own reference to the frozen values, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: RestrictedFunction.__init__() got an unexpected keyword argument 'restriction_indices'`
    - **See** issue [#1671](https://gitlab.com/gemseo/dev/gemseo/-/issues/1671), merge request [!2509](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2509).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.core.mdo_functions.restricted_function import RestrictedFunction
    function = MDOFunction(lambda x: np.array([x[0] + 2 * x[1] + 3 * x[2]]), "f")
    restricted = RestrictedFunction(function, restriction_indices=np.array([1]), restriction_values=np.array([10.0]))
    print([restricted.evaluate(np.array([1.0, 1.0])), restricted.restriction_values])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.core.function.restricted_function import RestrictedFunction
    function = ArrayFunction(lambda x: np.array([x[0] + 2 * x[1] + 3 * x[2]]), "f")
    frozen_values = np.array([10.0])
    restricted = RestrictedFunction(function, frozen_indexes=np.array([1]), frozen_values=frozen_values)
    print([restricted.evaluate(np.array([1.0, 1.0])), frozen_values])
    ```

### `ScalabilityStudy.add_optimization_strategy` takes settings models

In GEMSEO 7, the `algo_name` and `max_iter` arguments of `ScalabilityStudy.add_optimization_strategy` are replaced by `algo_settings` of `BaseOptimizerSettings` type, and its `formulation_name` argument by `formulation_settings` of `BaseFormulationSettings` type. Consequently, the items of `algorithms_options` are settings models, read by attribute instead of by key.

bump-gemseo converts the calls to `add_optimization_strategy` but leaves no comment on the reads of `algorithms_options`, so `settings["max_iter"]` raises `TypeError: 'SLSQP_Settings' object is not subscriptable`. Replace the item access by an attribute access, `settings.max_iter`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: 'SLSQP_Settings' object is not subscriptable`
    - **See** merge request [!2403](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2403).

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.mdo.scalable.data_driven.study.process import ScalabilityStudy
    study = ScalabilityStudy("y", ["x"], directory="study_dir")
    study.add_optimization_strategy("SLSQP", 15)
    study.add_optimization_strategy("NLOPT_COBYLA", 20, formulation_name="MDF")
    print([list(study.algorithms), list(study.formulations), [settings["max_iter"] for settings in study.algorithms_options]])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.formulation import MDF_Settings
    from gemseo.optimization import NLOPT_COBYLA_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.problem.mdo.scalable.data_driven.study.process import ScalabilityStudy
    study = ScalabilityStudy("y", ["x"], directory="study_dir")
    study.add_optimization_strategy(SLSQP_Settings(max_iter=15), DisciplinaryOpt_Settings())
    study.add_optimization_strategy(NLOPT_COBYLA_Settings(max_iter=20), MDF_Settings())
    print([list(study.algorithms), list(study.formulations), [settings.max_iter for settings in study.algorithms_options]])
    ```

### Renamed argument of `SobieskiProblem.get_constraints`

The `design_vector` argument of `SobieskiProblem.get_constraints` is renamed `input_value` in GEMSEO 7, and the module `gemseo.problems.mdo.sobieski.core` is now `gemseo.problem.mdo.sobieski.standalone`.

bump-gemseo leaves no comment on the argument, so a call with the old keyword raises `TypeError: SobieskiProblem.get_constraints() got an unexpected keyword argument 'design_vector'`. Rename the keyword argument, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: SobieskiProblem.get_constraints() got an unexpected keyword argument 'design_vector'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
    problem = SobieskiProblem()
    constraints = problem.get_constraints(design_vector=problem.initial_design)
    print([len(constraints), constraints[0]])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.problem.mdo.sobieski.standalone.problem import SobieskiProblem
    problem = SobieskiProblem()
    constraints = problem.get_constraints(input_value=problem.initial_design)
    print([len(constraints), constraints[0]])
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### The data-driven `ScalableProblem.create_scenario` takes settings models

For data-driven scalable problems, GEMSEO 7 replaces the `formulation_name` and `**formulation_settings` arguments of `ScalableProblem.create_scenario` by `formulation_settings` of `BaseFormulationSettings` type and `sub_optimizer_settings` of `BaseOptimizerSettings` type, and removes its `scenario_type` argument, as the scenario is an `MDOScenario`.

bump-gemseo cannot tell the data-driven `ScalableProblem` from the parametric one, so it does not migrate the positional arguments; it writes a comment giving the rewriting for each case, and another one asking to remove `scenario_type`. Left as is, the call raises `TypeError: ScalableProblem.create_scenario() got an unexpected keyword argument 'scenario_type'`. Follow the data-driven case: pass the formulation as a settings model, e.g. `formulation_settings=MDF_Settings()`, and remove `scenario_type`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: the griffe dump has several signatures of ScalableProblem.create_scenario, so the positional arguments are not migrated: for gemseo.problems.mdo.scalable.data_driven.problem.ScalableProblem, write problem.create_scenario(scenario_type="DOE", start_at_equilibrium=True, formulation_settings=MDF_Settings()), then see the TODO: remove it, as the scenario is an MDOScenario; for gemseo.problems.mdo.scalable.parametric.scalable_problem.ScalableProblem, keep problem.create_scenario("MDF", scenario_type="DOE", start_at_equilibrium=True), then see the TODOs: cannot transform: scenario_type, start_at_equilibrium are no fields of MDF_Settings, so the call keeps them; migrate by hand; remove it, as the scenario is an MDOScenario` / `remove it, as the scenario is an MDOScenario`
    - **Error with GEMSEO 7**: `TypeError: ScalableProblem.create_scenario() got an unexpected keyword argument 'scenario_type'`
    - **See** merge request [!2403](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2403).

=== "GEMSEO 6.3"

    ```python
    from gemseo import create_discipline
    from gemseo import create_scenario
    from gemseo.problems.mdo.aerostructure.aerostructure_design_space import AerostructureDesignSpace
    from gemseo.problems.mdo.scalable.data_driven.problem import ScalableProblem
    disciplines = create_discipline(["Aerodynamics", "Structure", "Mission"])
    datasets = []
    for discipline in disciplines:
        design_space = AerostructureDesignSpace()
        design_space.filter(discipline.io.input_grammar.names)
        output_names = iter(discipline.io.output_grammar.names)
        sampling = create_scenario(discipline, next(output_names), design_space, formulation_name="DisciplinaryOpt", scenario_type="DOE")
        for output_name in output_names:
            sampling.add_observable(output_name)
        sampling.execute(algo_name="DiagonalDOE", n_samples=10)
        datasets.append(sampling.to_dataset(name=discipline.name, opt_naming=False))
    problem = ScalableProblem(datasets, ["thick_airfoils", "thick_panels", "sweep"], "range", ["c_rf"], ["c_lift"], True, sizes={"sweep": 2})
    scenario = problem.create_scenario("MDF", scenario_type="DOE", start_at_equilibrium=True)
    print([scenario.formulation.__class__.__name__, sorted(scenario.design_space.variable_names)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo import create_discipline
    from gemseo import create_scenario
    from gemseo.doe import DiagonalDOE_Settings
    from gemseo.formulation import MDF_Settings
    from gemseo.problem.mdo.aerostructure import AerostructureDesignSpace
    from gemseo.problem.mdo.scalable.data_driven import ScalableProblem
    disciplines = create_discipline(["Aerodynamics", "Structure", "Mission"])
    datasets = []
    for discipline in disciplines:
        design_space = AerostructureDesignSpace()
        design_space.filter(discipline.io.input_grammar.names)
        output_names = iter(discipline.io.output_grammar.names)
        sampling = create_scenario(discipline, next(output_names), design_space, formulation_name="DisciplinaryOpt", scenario_type="DOE")
        for output_name in output_names:
            sampling.add_observable(output_name)
        sampling.execute(DiagonalDOE_Settings(n_samples=10))
        datasets.append(sampling.to_dataset(name=discipline.name, opt_naming=False))
    problem = ScalableProblem(datasets, ["thick_airfoils", "thick_panels", "sweep"], "range", ["c_rf"], ["c_lift"], True, sizes={"sweep": 2})
    scenario = problem.create_scenario(formulation_settings=MDF_Settings(), start_at_equilibrium=True)
    print([scenario.formulation.__class__.__name__, sorted(scenario.design_space.variables)])
    ```

### `EvaluationProblem.preprocess_functions` was replaced by `bind_functions`

`EvaluationProblem.preprocess_functions()` and the `preprocessing` argument of `EvaluationProblem.reset()` are removed in GEMSEO 7. `EvaluationProblem.bind_functions()` binds the functions of a problem to its database and counters, and `EvaluationProblem.create_working_problem(transformation)` returns a new problem whose functions work in the coordinates of a space transformation. Code that only solves problems needs no change.

bump-gemseo removes the `preprocessing` argument and marks the call to `preprocess_functions` with a comment pointing to these two methods; left as is, the call raises an `AttributeError`. If you only evaluate the functions, remove the call, as in the example below; otherwise call `bind_functions` or `create_working_problem`, depending on what you need. Note also that the `design_vector_is_normalized` argument of `EvaluationProblem.evaluate_functions` is renamed `input_value_is_normalized`.

!!! info "After bump-gemseo"

    - **Comment**: `use bind_functions to bind the functions to the database and the counters, and create_working_problem to derive a problem whose functions work in the coordinates an algorithm manipulates`
    - **Error with GEMSEO 7**: `AttributeError: 'EvaluationProblem' object has no attribute 'preprocess_functions'`

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.evaluation_problem import EvaluationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=2.0, value=1.0)
    problem = EvaluationProblem(space)
    problem.add_observable(MDOFunction(lambda x: 2 * x, "t"))
    problem.preprocess_functions(is_function_input_normalized=False)
    problem.reset(preprocessing=True)
    values, _ = problem.evaluate_functions(np.array([1.5]), design_vector_is_normalized=False, output_functions=problem.observables)
    print(values["t"])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.doe import EvaluationProblem
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=2.0, value=1.0)
    problem = EvaluationProblem(space)
    problem.add_observable(ArrayFunction(lambda x: 2 * x, "t"))
    values, _ = problem.evaluate_functions(np.array([1.5]), input_value_is_normalized=False, output_functions=problem.observables)
    print(values["t"])
    ```

### `MDOFunction.check_grad` was replaced by `FunctionJacobianChecker`

`MDOFunction.check_grad()` is removed in GEMSEO 7, where `MDOFunction` is renamed `ArrayFunction`; use `gemseo.util.derivative.check.function.FunctionJacobianChecker` instead, which checks the Jacobian of an `ArrayFunction`. bump-gemseo marks the call with a comment saying so; left as is, it raises `AttributeError: 'ArrayFunction' object has no attribute 'check_grad'`.

Create a `FunctionJacobianChecker` from the function and call its `check` method with the input value, as in the example below, where the `error_max` argument becomes `atol`. While `check_grad` raised a `ValueError` for a wrong Jacobian, `check` returns a value that is false in that case, so replace a `try`/`except` block by a test of this value.

!!! info "After bump-gemseo"

    - **Comment**: `use gemseo.util.derivative.check.function.FunctionJacobianChecker instead`
    - **Error with GEMSEO 7**: `AttributeError: 'ArrayFunction' object has no attribute 'check_grad'`
    - **See** issue [#747](https://gitlab.com/gemseo/dev/gemseo/-/issues/747), merge request [!2511](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2511).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    function = MDOFunction(lambda x: np.array([x[0] ** 2 + x[1]]), "f", jac=lambda x: np.array([[2 * x[0], 1.0]]))
    function.check_grad(np.array([1.0, 2.0]), step=1e-7, error_max=1e-5)
    wrong = MDOFunction(lambda x: np.array([x[0] ** 2]), "g", jac=lambda x: np.array([[1.0]]))
    try:
        wrong.check_grad(np.array([3.0]), error_max=1e-5)
        detected = False
    except ValueError:
        detected = True
    print([True, detected])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.util.derivative.check.function import FunctionJacobianChecker
    function = ArrayFunction(lambda x: np.array([x[0] ** 2 + x[1]]), "f", jac=lambda x: np.array([[2 * x[0], 1.0]]))
    first = FunctionJacobianChecker(function).check(np.array([1.0, 2.0]), step=1e-7, atol=1e-5)
    wrong = ArrayFunction(lambda x: np.array([x[0] ** 2]), "g", jac=lambda x: np.array([[1.0]]))
    detected = not FunctionJacobianChecker(wrong).check(np.array([3.0]), atol=1e-5)
    print([first, detected])
    ```

### `MDOLinearFunction.normalize` was removed

In GEMSEO 7, `MDOLinearFunction` is renamed `LinearFunction` and its `normalize` method, which took a design space, is removed without a direct replacement: normalizing a design space is a transformation of the space, see `gemseo.space.transformation`.

bump-gemseo marks the call with the comment `MDOLinearFunction.normalize was removed; see the GEMSEO 7 changelog.`; left as is, it raises `AttributeError: 'LinearFunction' object has no attribute 'normalize'`. Remove the call and the code using the normalized function.

!!! info "After bump-gemseo"

    - **Comment**: `MDOLinearFunction.normalize was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'LinearFunction' object has no attribute 'normalize'`

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction
    space = DesignSpace()
    space.add_variable("x", size=2, lower_bound=0.0, upper_bound=2.0, value=1.0)
    function = MDOLinearFunction(np.array([1.0, 2.0]), "f", value_at_zero=1.0)
    normalized = function.normalize(space)
    print(normalized.evaluate(np.array([0.5, 0.5])))
    ```

=== "GEMSEO 7"

    ```python
    print("<removed in GEMSEO 7>")
    ```

### `ProblemFunction` was replaced by `EvaluationFunction` and `TransformedInputFunction`

The `ProblemFunction` class of `gemseo.algos.problem_function` is removed in GEMSEO 7, together with `EvaluationProblem.preprocess_functions()`. Depending on what you used it for, replace it with `EvaluationFunction` to keep track of the evaluations of a function, with `TransformedInputFunction` to adapt its argument to the coordinates an algorithm works on, or with both stacked, as `EvaluationProblem.bind_functions` and `EvaluationProblem.create_working_problem` do.

bump-gemseo marks the import with a comment giving these options; left as is, the import raises an `ImportError` with the same message. As in the example below, `EvaluationFunction` is imported from `gemseo.core.function.evaluation_function` and derives from `ArrayFunction`, the new name of `MDOFunction`.

!!! info "After bump-gemseo"

    - **Comment**: `gemseo.algos.problem_function.ProblemFunction: replace it with EvaluationFunction to record the evaluations of a function, TransformedInputFunction to adapt its argument to the coordinates an algorithm works on, or both stacked as EvaluationProblem.bind_functions and EvaluationProblem.create_working_problem do`
    - **Error with GEMSEO 7**: `ImportError: The attribute 'ProblemFunction' of the module 'gemseo.algos.problem_function' was removed; replace it with EvaluationFunction to record the evaluations of a function, TransformedInputFunction to adapt its ar`
    - **See** issue [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846), merge request [!2579](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2579).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.problem_function import ProblemFunction
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    print(issubclass(ProblemFunction, MDOFunction))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.core.function.evaluation_function import EvaluationFunction
    print(issubclass(EvaluationFunction, ArrayFunction))
    ```

### `EvaluationProblem._functions_are_preprocessed` was removed

The private flag `EvaluationProblem._functions_are_preprocessed` was removed in GEMSEO 7 with `preprocess_functions`. Its replacement, `EvaluationProblem.bind_functions()`, needs no such flag, as it rebuilds the wrappers from the original functions when called again.

bump-gemseo marks the line reading the flag and the call to `preprocess_functions` with comments; left as is, reading the flag raises an `AttributeError`. Remove the code that reads the flag and, if you need to bind the functions of the problem to its database and counters, call `bind_functions`, possibly several times.

!!! info "After bump-gemseo"

    - **Comment**: `the flag was removed with preprocess_functions; bind_functions needs none, as it rebuilds the wrappers from the original functions when called again` / `use bind_functions to bind the functions to the database and the counters, and create_working_problem to derive a problem whose functions work in the coordinates an algorithm manipulates`
    - **Error with GEMSEO 7**: `AttributeError: 'OptimizationProblem' object has no attribute '_functions_are_preprocessed'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    problem = OptimizationProblem(space)
    problem.objective = MDOFunction(lambda x: x, "f")
    before = problem._functions_are_preprocessed
    problem.preprocess_functions()
    print([before, problem._functions_are_preprocessed])
    ```

=== "GEMSEO 7"

    ```python
    print("<removed in GEMSEO 7>")
    ```

### `ScalableProblem.create_scenario` no longer takes `use_optimizer`

For parametric scalable problems, the `use_optimizer` argument of `ScalableProblem.create_scenario` is removed in GEMSEO 7, and the scenario it creates is an `MDOScenario`. In the same method, the `formulation_name` and `**formulation_settings` arguments are replaced by `formulation_settings: BaseFormulationSettings | None = None`.

bump-gemseo marks the call with the comment `remove it, as the scenario is an MDOScenario`; left as is, it raises `TypeError: ScalableProblem.create_scenario() got an unexpected keyword argument 'use_optimizer'`. Remove the argument, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `remove it, as the scenario is an MDOScenario`
    - **Error with GEMSEO 7**: `TypeError: ScalableProblem.create_scenario() got an unexpected keyword argument 'use_optimizer'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.mdo.scalable.parametric.scalable_problem import ScalableProblem
    scenario = ScalableProblem().create_scenario(use_optimizer=False)
    print([scenario.__class__.__name__.replace("DOE", "MDO"), sorted(scenario.design_space.variable_names)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.problem.mdo.scalable.parametric import ScalableProblem
    scenario = ScalableProblem().create_scenario()
    print([scenario.__class__.__name__, sorted(scenario.design_space.variables)])
    ```

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. Unless an entry above gives an equivalent, they have no replacement: remove the code that uses them.

- `FunctionFromDiscipline.generator_class`
- `MDOFunction.filt_0`
- `MDOFunction.rel_err`
- `gemseo.core.mdo_functions.mdo_function.OperatorType`
- `gemseo.problems.mdo.sobieski.process.mda_gauss_seidel`
- `gemseo.problems.mdo.sobieski.process.mda_jacobi`
