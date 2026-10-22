---
description: "Migrating optimization from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Optimization

The changes of GEMSEO 7 about optimization libraries and their settings that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### The augmented Lagrangian algorithms are renamed with `Order`

In GEMSEO 7, the algorithms `"Augmented_Lagrangian_order_0"` and `"Augmented_Lagrangian_order_1"` are renamed `"Augmented_Lagrangian_Order_0"` and `"Augmented_Lagrangian_Order_1"`, like their settings classes `Augmented_Lagrangian_Order_0_Settings` and `Augmented_Lagrangian_Order_1_Settings`.

bump-gemseo does not rename these strings and leaves no comment, so the old name raises `KeyError: 'The algorithm Augmented_Lagrangian_order_0 is unknown in AugmentedLagrangianOrder0; available ones are: Augmented_Lagrangian_Order_0.'`. Use the new names and pass the settings, including the sub-algorithm, as a settings model, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `KeyError: 'The algorithm Augmented_Lagrangian_order_0 is unknown in AugmentedLagrangianOrder0; available ones are: Augmented_Lagrangian_Order_0.'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.opt.augmented_lagrangian.augmented_lagrangian_order_0 import AugmentedLagrangianOrder0
    from gemseo.algos.opt.augmented_lagrangian.augmented_lagrangian_order_1 import AugmentedLagrangianOrder1
    from gemseo.problems.optimization.power_2 import Power2
    first = AugmentedLagrangianOrder0("Augmented_Lagrangian_order_0").execute(Power2(), sub_algorithm_name="SLSQP", max_iter=100)
    second = AugmentedLagrangianOrder1("Augmented_Lagrangian_order_1").execute(Power2(), sub_algorithm_name="SLSQP", max_iter=100)
    print([first.f_opt, second.f_opt])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.optimization import Augmented_Lagrangian_Order_0_Settings
    from gemseo.optimization import Augmented_Lagrangian_Order_1_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.optimization.augmented_lagrangian.order_0 import AugmentedLagrangianOrder0
    from gemseo.optimization.augmented_lagrangian.order_1 import AugmentedLagrangianOrder1
    from gemseo.problem.optimization.power_2 import Power2
    first = AugmentedLagrangianOrder0("Augmented_Lagrangian_Order_0").execute(Power2(), settings=Augmented_Lagrangian_Order_0_Settings(sub_algorithm_settings=SLSQP_Settings(), max_iter=100))
    second = AugmentedLagrangianOrder1("Augmented_Lagrangian_Order_1").execute(Power2(), settings=Augmented_Lagrangian_Order_1_Settings(sub_algorithm_settings=SLSQP_Settings(), max_iter=100))
    print([first.f_opt, second.f_opt])
    ```

### The `disp` and `iprint` settings of `L_BFGS_B` were removed

The `disp` and `iprint` options of the L-BFGS-B optimizer (`L_BFGS_B_Settings`) are removed in GEMSEO 7, as they are no longer available with SciPy 1.15 or later, which deprecated them for this algorithm; the `disp` setting of the other SciPy local optimizers is removed too.

bump-gemseo leaves no comment, and the settings model rejects them with a `ValidationError` (`Extra inputs are not permitted`). Remove these settings, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ValidationError: 2 validation errors for L_BFGS_B_Settings iprint Extra inputs are not permitted [type=extra_forbidden, input_value=-1, input_type=int] For further information visit https://errors.pydantic.dev/2.14/v/ext`
    - **See** issue [#1853](https://gitlab.com/gemseo/dev/gemseo/-/issues/1853), merge request [!2635](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2635).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    from gemseo.algos.opt.scipy_local.settings.lbfgsb import L_BFGS_B_Settings
    from gemseo.problems.optimization.rosenbrock import Rosenbrock
    result = OptimizationLibraryFactory().execute(Rosenbrock(), settings_model=L_BFGS_B_Settings(max_iter=200, iprint=-1, disp=False))
    print(result.f_opt < 1e-6)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.optimization import L_BFGS_B_Settings
    from gemseo.optimization.factory import OptimizationLibraryFactory
    from gemseo.problem.optimization.rosenbrock import Rosenbrock
    result = OptimizationLibraryFactory().execute(Rosenbrock(), settings=L_BFGS_B_Settings(max_iter=200))
    print(result.f_opt < 1e-6)
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### `execute` of an algorithm library takes a settings model

In GEMSEO 7, the `**settings` arguments of `BaseAlgorithmLibrary.execute` are removed and its `settings_model` argument is renamed `settings`, still optional; the same holds for `BaseDriverLibrary.execute`. The `eval_obs_jac` and `max_design_space_dimension_to_log` arguments of `BaseDriverLibrary.execute` are replaced by the `evaluate_observable_jacobian` field of `BaseDriverSettings` and the `max_input_space_dimension_to_log` field of `BaseDriverLibrarySettings`.

bump-gemseo cannot gather the settings into a settings model, as `eval_obs_jac` and `max_design_space_dimension_to_log` are no fields of `SLSQP_Settings`; it keeps the call and marks it with a comment. Left as is, the call raises `TypeError: BaseDriverLibrary.execute() got an unexpected keyword argument 'max_iter'`. Gather all the settings, with the new field names, into the settings model of the algorithm, pass it as `settings`, and rename `settings_model` to `settings`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: eval_obs_jac, max_design_space_dimension_to_log are no fields of SLSQP_Settings, so the call keeps them; migrate by hand`
    - **Error with GEMSEO 7**: `TypeError: BaseDriverLibrary.execute() got an unexpected keyword argument 'max_iter'`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2344](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2344); issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2348](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2348); merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
    from gemseo.problems.optimization.rosenbrock import Rosenbrock
    library = OptimizationLibraryFactory().create("SLSQP")
    problem = Rosenbrock()
    first = library.execute(problem, max_iter=100, eval_obs_jac=False, max_design_space_dimension_to_log=10)
    second = library.execute(Rosenbrock(), settings_model=SLSQP_Settings(max_iter=50))
    print([first.f_opt, second.f_opt])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.optimization import SLSQP_Settings
    from gemseo.optimization.factory import OptimizationLibraryFactory
    from gemseo.problem.optimization.rosenbrock import Rosenbrock
    library = OptimizationLibraryFactory().create("SLSQP")
    problem = Rosenbrock()
    first = library.execute(problem, settings=SLSQP_Settings(max_iter=100, evaluate_observable_jacobian=False, max_input_space_dimension_to_log=10))
    second = library.execute(Rosenbrock(), settings=SLSQP_Settings(max_iter=50))
    print([first.f_opt, second.f_opt])
    ```

### The algorithm `Scipy_MILP` is renamed `MILP`

The algorithm `"Scipy_MILP"` is named `"MILP"` in GEMSEO 7, with the settings class `MILP_Settings`. Besides, the factory of the optimization libraries no longer takes the algorithm name and its settings as keyword arguments, but a settings model.

bump-gemseo cannot convert the call, as there is no `Scipy_MILP_Settings` class, and marks it with a comment; left as is, the call raises `TypeError: BaseAlgorithmFactory.execute() got an unexpected keyword argument 'algo_name'`. Replace `algo_name="Scipy_MILP"` and the other settings by `settings=MILP_Settings()`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: no Settings class Scipy_MILP_Settings in the griffe dump to gather the settings into`
    - **Error with GEMSEO 7**: `TypeError: BaseAlgorithmFactory.execute() got an unexpected keyword argument 'algo_name'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction
    import numpy as np
    space = DesignSpace()
    space.add_variable("x", size=2, type_="integer", lower_bound=0, upper_bound=3, value=0)
    problem = OptimizationProblem(space)
    problem.objective = MDOLinearFunction(np.array([-1.0, -2.0]), "f")
    problem.add_constraint(MDOLinearFunction(np.array([1.0, 1.0]), "g", value_at_zero=-4.0), constraint_type="ineq")
    result = OptimizationLibraryFactory().execute(problem, algo_name="Scipy_MILP")
    print([result.f_opt, result.x_opt])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.core.function.linear_function import LinearFunction
    from gemseo.optimization import MILP_Settings
    from gemseo.optimization import OptimizationProblem
    from gemseo.optimization.factory import OptimizationLibraryFactory
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", size=2, type_="integer", lower_bound=0, upper_bound=3, value=0)
    problem = OptimizationProblem(space)
    problem.objective = LinearFunction(np.array([-1.0, -2.0]), "f")
    problem.add_constraint(LinearFunction(np.array([1.0, 1.0]), "g", value_at_zero=-4.0), constraint_type="ineq")
    result = OptimizationLibraryFactory().execute(problem, settings=MILP_Settings())
    print([result.f_opt, result.x_opt])
    ```

### The sub-algorithm of an augmented Lagrangian is a settings model

In GEMSEO 7, `BaseAugmentedLagragianSettings` is renamed `BaseAugmentedLagrangianSettings` and its `sub_algorithm_name` field is removed: the sub-algorithm is given by the `sub_algorithm_settings` field, now of type `BaseOptimizerSettings`. The settings class `Augmented_Lagrangian_order_0_Settings` is also renamed `Augmented_Lagrangian_Order_0_Settings`.

bump-gemseo marks the reads of `sub_algorithm_name` with a comment; left as is, they raise an `AttributeError`. Pass the settings model of the sub-algorithm, e.g. `SLSQP_Settings(max_iter=50)`, as `sub_algorithm_settings`, and read the name of the sub-algorithm from the class of this model, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `BaseAugmentedLagragianSettings.sub_algorithm_name was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'Augmented_Lagrangian_Order_0_Settings' object has no attribute 'sub_algorithm_name'`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2348](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2348).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.opt.augmented_lagrangian.augmented_lagrangian_order_0 import AugmentedLagrangianOrder0
    from gemseo.algos.opt.augmented_lagrangian.settings.augmented_lagrangian_order_0_settings import Augmented_Lagrangian_order_0_Settings
    from gemseo.problems.optimization.power_2 import Power2
    settings = Augmented_Lagrangian_order_0_Settings(sub_algorithm_name="SLSQP", sub_algorithm_settings={"max_iter": 50}, max_iter=200)
    result = AugmentedLagrangianOrder0().execute(Power2(), settings_model=settings)
    print([result.f_opt, settings.sub_algorithm_name])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.optimization import Augmented_Lagrangian_Order_0_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.optimization.augmented_lagrangian.order_0 import AugmentedLagrangianOrder0
    from gemseo.problem.optimization.power_2 import Power2
    settings = Augmented_Lagrangian_Order_0_Settings(sub_algorithm_settings=SLSQP_Settings(max_iter=50), max_iter=200)
    result = AugmentedLagrangianOrder0().execute(Power2(), settings=settings)
    print([result.f_opt, settings.sub_algorithm_settings.__class__.__name__.removesuffix("_Settings")])
    ```

### Offset and sign are ignored for a function created as a constraint

In GEMSEO 7, if the function passed to `OptimizationProblem.add_constraint` was created with a constraint type (`ConstraintType.EQ` or `INEQ` as its function type), the `value` and `positive` arguments are now ignored without warning.

bump-gemseo marks the call with a comment, but the code runs without error and gives other constraint values. Check the function type of the constrained function (`f_type`): either create the function with the default function type and pass the type with the `constraint_type` argument of `add_constraint`, as in the example below, or apply the offset and the sign to the function itself.

!!! info "After bump-gemseo"

    - **Comment**: `if the function was created with a constraint type (ConstraintType.EQ or INEQ as its function type), value and positive are now ignored without warning; create it with the default function type, or apply the offset and the sign to the function itself`
    - **With GEMSEO 7**: no error, but other results.

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=4.0, value=1.0)
    problem = OptimizationProblem(space)
    problem.objective = MDOFunction(lambda x: x, "f")
    constraint = MDOFunction(lambda x: x, "g", f_type=MDOFunction.FunctionType.INEQ)
    problem.add_constraint(constraint, value=1.0, positive=True)
    print([c.evaluate(np.array([3.0])) for c in problem.constraints])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.optimization import OptimizationProblem
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=4.0, value=1.0)
    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(lambda x: x, "f")
    constraint = ArrayFunction(lambda x: x, "g")
    problem.add_constraint(constraint, value=1.0, constraint_type=ArrayFunction.ConstraintType.INEQ, positive=True)
    print([c.evaluate(np.array([3.0])) for c in problem.constraints])
    ```

### Renamed fields of the driver settings

In GEMSEO 7, the driver settings `eval_obs_jac` and `max_design_space_dimension_to_log` are renamed `evaluate_observable_jacobian` and `max_input_space_dimension_to_log`. Besides, the `**settings` arguments of `BaseAlgoFactory.execute`, renamed `BaseAlgorithmFactory`, are removed, and its `settings` argument, formerly `settings_model`, is now mandatory.

bump-gemseo cannot gather the settings into a settings model, as the old names are no fields of `SLSQP_Settings`; it keeps the call and marks it with a comment. Left as is, the call raises `TypeError: BaseAlgorithmFactory.execute() got an unexpected keyword argument 'eval_obs_jac'`. Create the settings model of the algorithm with the new field names, pass it as `settings` and remove `algo_name`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: eval_obs_jac, max_design_space_dimension_to_log are no fields of SLSQP_Settings, so the call keeps them; migrate by hand`
    - **Error with GEMSEO 7**: `TypeError: BaseAlgorithmFactory.execute() got an unexpected keyword argument 'eval_obs_jac'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    from gemseo.problems.optimization.rosenbrock import Rosenbrock
    result = OptimizationLibraryFactory().execute(Rosenbrock(), algo_name="SLSQP", max_iter=80, eval_obs_jac=True, max_design_space_dimension_to_log=5)
    print(result.f_opt)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.optimization import SLSQP_Settings
    from gemseo.optimization.factory import OptimizationLibraryFactory
    from gemseo.problem.optimization.rosenbrock import Rosenbrock
    result = OptimizationLibraryFactory().execute(Rosenbrock(), settings=SLSQP_Settings(max_iter=80, evaluate_observable_jacobian=True, max_input_space_dimension_to_log=5))
    print(result.f_opt)
    ```

### `skip_int_check` and `round_ints` were replaced by integer relaxation

In GEMSEO 7, the driver setting `round_ints` and the `skip_int_check` argument of `BaseDriverLibrary.execute()` are removed. They are replaced by the driver settings `relax_integer_variables` and `relax_discrete_variables`, `False` by default, which have another meaning: an algorithm that does not handle integer or discrete variables raises a `ValueError` on them unless they are relaxed to float variables, and nothing rounds the point at every evaluation any more.

bump-gemseo keeps `skip_int_check` in the call and marks it, as well as the reads of `round_ints`, with comments; left as is, the call raises a `TypeError`. Remove these arguments and fields, as in the example below. If your problem has integer variables that the algorithm does not handle, set `relax_integer_variables=True` and check the results: in a relaxed run, `OptimizationResult.x_opt` is relaxed and its projection onto the declared domain is given by `OptimizationResult.x_opt_projected`.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: skip_int_check is no field of SLSQP_Settings, so the call keeps it; migrate by hand` / `BaseDriverSettings.round_ints was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `TypeError: BaseDriverLibrary.execute() got an unexpected keyword argument 'max_iter'`
    - **GEMSEO 7 cannot give the GEMSEO 6.3 value here**: the driver setting round_ints and the skip_int_check argument are replaced by relax_integer_variables, with another meaning (changelog 1528).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
    from gemseo.problems.optimization.rosenbrock import Rosenbrock
    library = OptimizationLibraryFactory().create("SLSQP")
    result = library.execute(Rosenbrock(), max_iter=100, skip_int_check=True)
    other = OptimizationLibraryFactory().execute(Rosenbrock(), algo_name="SLSQP", max_iter=50, skip_int_check=False)
    settings = SLSQP_Settings(round_ints=False)
    print([result.f_opt, other.f_opt, settings.round_ints])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.optimization import SLSQP_Settings
    from gemseo.optimization.factory import OptimizationLibraryFactory
    from gemseo.problem.optimization.rosenbrock import Rosenbrock
    library = OptimizationLibraryFactory().create("SLSQP")
    result = library.execute(Rosenbrock(), settings=SLSQP_Settings(max_iter=100))
    other = OptimizationLibraryFactory().execute(Rosenbrock(), settings=SLSQP_Settings(max_iter=50))
    print([result.f_opt, other.f_opt, "<removed in GEMSEO 7>"])
    ```

### `BaseAlgorithmLibrary._filter_settings` takes no argument

In GEMSEO 7, the `_filter_settings` method of `BaseAlgorithmLibrary` takes no argument: use `self._filter_settings()` instead of `self._filter_settings(settings, model_to_exclude)`. It dumps the settings of the library by field name, not by alias, without the fields of the class attributes `_settings_class_to_exclude` and `_additional_settings_classes_to_exclude`, which a library sets instead of passing a model to exclude.

bump-gemseo marks the call with two comments; left as is, it raises a `TypeError`. If the mapping you passed was not `self._settings.model_dump()`, filter it yourself; if `model_to_exclude` was another class, set these class attributes. Outside a library, call the method on an instance whose `_settings` is set, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `_filter_settings() now dumps self._settings by field name, not by alias; filter the mapping yourself if it was not self._settings.model_dump()` / `_filter_settings() now excludes the fields of the classes of _settings_class_to_exclude and _additional_settings_classes_to_exclude; set these class attributes if model_to_exclude was another class`
    - **Error with GEMSEO 7**: `TypeError: BaseAlgorithmLibrary._filter_settings() missing 1 required positional argument: 'self'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
    from gemseo.algos.opt.scipy_local.scipy_local import ScipyOpt
    from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
    filtered = ScipyOpt._filter_settings(SLSQP_Settings(max_iter=5).model_dump(), BaseOptimizerSettings)
    print(sorted(filtered))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.optimization import SLSQP_Settings
    from gemseo.optimization.scipy_local.scipy_local import ScipyOpt
    library = ScipyOpt("SLSQP")
    library._settings = SLSQP_Settings(max_iter=5)
    print(sorted(library._filter_settings()))
    ```

### `MNBI_Settings` takes nested settings models

In GEMSEO 7, the `sub_optim_algo`, `sub_optim_max_iter` and `doe_algo` fields of `MNBI_Settings` are removed. Use its `sub_optim_algo_settings` field, of type `BaseOptimizerSettings`, for the sub-optimizer and its maximum number of iterations, and its `doe_algo_settings` field, of type `BaseDOESettings`, for the DOE algorithm.

bump-gemseo marks the reads of the removed fields with comments such as `MNBI_Settings.sub_optim_algo was removed; see the GEMSEO 7 changelog.`; left as is, they raise an `AttributeError`. Build the nested settings models, as in the example below, and read the values from them; the name of an algorithm is that of the class of its settings model without the suffix `_Settings`.

!!! info "After bump-gemseo"

    - **Comment**: `MNBI_Settings.sub_optim_algo was removed; see the GEMSEO 7 changelog.` / `MNBI_Settings.sub_optim_max_iter was removed; see the GEMSEO 7 changelog.` / `MNBI_Settings.doe_algo was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'MNBI_Settings' object has no attribute 'sub_optim_algo'`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2348](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2348).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.opt.mnbi.settings.mnbi_settings import MNBI_Settings
    settings = MNBI_Settings(n_sub_optim=3, sub_optim_algo="SLSQP", sub_optim_max_iter=20, doe_algo="PYDOE_FULLFACT", max_iter=100)
    print([settings.sub_optim_algo, settings.sub_optim_max_iter, settings.doe_algo])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import PYDOE_FULLFACT_Settings
    from gemseo.optimization import MNBI_Settings
    from gemseo.optimization import SLSQP_Settings
    settings = MNBI_Settings(n_sub_optim=3, sub_optim_algo_settings=SLSQP_Settings(max_iter=20), doe_algo_settings=PYDOE_FULLFACT_Settings(), max_iter=100)
    print([settings.sub_optim_algo_settings.__class__.__name__.removesuffix("_Settings"), settings.sub_optim_algo_settings.max_iter, settings.doe_algo_settings.__class__.__name__.removesuffix("_Settings")])
    ```

### `MultiStart_Settings` takes nested settings models

In GEMSEO 7, the `opt_algo_name`, `opt_algo_max_iter`, `doe_algo_name` and `n_start` fields of `MultiStart_Settings` are removed. Use its `opt_algo_settings` field, of type `BaseOptimizerSettings`, for the optimizer and its maximum number of iterations, and its `doe_algo_settings` field, of type `BaseDOESettings`, for the DOE algorithm and the number of starting points, with its `n_samples` or `samples` field for example.

bump-gemseo marks the reads of the removed fields with comments such as `MultiStart_Settings.n_start was removed; see the GEMSEO 7 changelog.`; left as is, they raise an `AttributeError`. Build the two nested settings models, as in the example below, and read the values from them, e.g. `settings.doe_algo_settings.n_samples` instead of `settings.n_start`. Pass the whole model with `settings` instead of `settings_model`.

!!! info "After bump-gemseo"

    - **Comment**: `MultiStart_Settings.n_start was removed; see the GEMSEO 7 changelog.` / `MultiStart_Settings.opt_algo_max_iter was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'MultiStart_Settings' object has no attribute 'n_start'`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2348](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2348).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    from gemseo.algos.opt.multi_start.settings.multi_start_settings import MultiStart_Settings
    from gemseo.problems.optimization.rosenbrock import Rosenbrock
    settings = MultiStart_Settings(opt_algo_name="SLSQP", opt_algo_max_iter=50, doe_algo_name="PYDOE_FULLFACT", n_start=4, max_iter=1000, n_processes=1)
    result = OptimizationLibraryFactory().execute(Rosenbrock(), settings_model=settings)
    print([result.f_opt < 1e-4, settings.n_start, settings.opt_algo_max_iter])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import PYDOE_FULLFACT_Settings
    from gemseo.optimization import MultiStart_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.optimization.factory import OptimizationLibraryFactory
    from gemseo.problem.optimization.rosenbrock import Rosenbrock
    settings = MultiStart_Settings(opt_algo_settings=SLSQP_Settings(max_iter=50), doe_algo_settings=PYDOE_FULLFACT_Settings(n_samples=4), max_iter=1000, n_processes=1)
    result = OptimizationLibraryFactory().execute(Rosenbrock(), settings=settings)
    print([result.f_opt < 1e-4, settings.doe_algo_settings.n_samples, settings.opt_algo_settings.max_iter])
    ```

### `_TARGET_CLASS_NAME` was replaced by `target_class_name`

`_TARGET_CLASS_NAME` of the settings classes is removed in GEMSEO 7: the name of the class a settings class targets is now derived from its own name, `X_Settings` targeting `X`, and given by the `target_class_name` property of `BaseSettings`.

bump-gemseo marks the line with a comment; left as is, it raises `AttributeError: _TARGET_CLASS_NAME`. As `target_class_name` is a property, read it from an instance of the settings class, as in the example below, or, for a class, remove the suffix `_Settings` from its name. If you defined `_TARGET_CLASS_NAME` in your own settings class, note that it is no longer read.

!!! info "After bump-gemseo"

    - **Comment**: `_TARGET_CLASS_NAME is no longer read, the target class name being derived from the name of the settings class (X_Settings targets X); read it with the target_class_name property of a settings instance, or, for a class, remove the suffix _Settings from its name`
    - **Error with GEMSEO 7**: `AttributeError: _TARGET_CLASS_NAME`

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
    print(SLSQP_Settings._TARGET_CLASS_NAME)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.optimization import SLSQP_Settings
    print(SLSQP_Settings().target_class_name)
    ```

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. Unless an entry above gives an equivalent, they have no replacement: remove the code that uses them.

- `gemseo.algos.opt.augmented_lagrangian.settings.base_augmented_lagrangian_settings.copy_field_opt`
- `gemseo.algos.opt.mnbi.mnbi.MNBIOptionsType`
- `gemseo.algos.opt.scipy_global.settings.base_scipy_global_settings.copy_field_opt`
- `gemseo.algos.opt.scipy_local.settings.base_scipy_local_settings.copy_field_opt`
