---
description: "Migrating datasets from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Datasets

The changes of GEMSEO 7 about datasets and their metadata that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Silent changes

The code runs after bump-gemseo, without error or comment, and gives other results.

### Dataset variable and group names keep insertion order

`Dataset.group_names`, `Dataset.variable_names`, `Dataset.variable_identifiers`, `Dataset.get_group_names` and `Dataset.get_variable_names` no longer return sorted values; they preserve insertion order. Likewise, `Dataset.variable_name_to_n_components` (formerly `variable_names_to_n_components`) uses the same order as the variables in the dataset.

After bump-gemseo, the code runs without error and without comment, but these names come in another order. Check the code that depends on this order, e.g. that indexes these lists, compares them or builds arrays from them, and sort the names explicitly with `sorted` where you need the GEMSEO 6.3 order, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1411](https://gitlab.com/gemseo/dev/gemseo/-/issues/1411), merge request [!2508](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2508); issue [#1718](https://gitlab.com/gemseo/dev/gemseo/-/issues/1718), merge request [!2312](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2312).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.dataset import Dataset
    dataset = Dataset()
    dataset.add_variable("zeta", np.array([[1.0], [2.0]]), group_name="outputs")
    dataset.add_variable("alpha", np.array([[3.0, 4.0], [5.0, 6.0]]), group_name="inputs")
    dataset.add_variable("beta", np.array([[7.0], [8.0]]), group_name="inputs")
    print([
        dataset.group_names, dataset.variable_names, [list(i) for i in dataset.variable_identifiers],
        dataset.get_group_names("alpha"), dataset.get_variable_names("inputs"), list(dataset.variable_names_to_n_components),
    ])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import Dataset
    dataset = Dataset()
    dataset.add_variable("zeta", np.array([[1.0], [2.0]]), group_name="outputs")
    dataset.add_variable("alpha", np.array([[3.0, 4.0], [5.0, 6.0]]), group_name="inputs")
    dataset.add_variable("beta", np.array([[7.0], [8.0]]), group_name="inputs")
    print([
        sorted(dataset.group_names), sorted(dataset.variable_names), sorted(list(i) for i in dataset.variable_identifiers),
        sorted(dataset.get_group_names("alpha")), sorted(dataset.get_variable_names("inputs")), sorted(dataset.variable_name_to_n_components),
    ])
    ```

### `IODataset` input and output names keep insertion order

`IODataset.input_names` and `IODataset.output_names` no longer return sorted values; they preserve insertion order.

After bump-gemseo, the code runs without error and without comment, but these names come in the order in which the variables were added. Check the code that depends on this order and sort the names explicitly with `sorted` where you need the GEMSEO 6.3 order, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1411](https://gitlab.com/gemseo/dev/gemseo/-/issues/1411), merge request [!2508](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2508).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.io_dataset import IODataset
    dataset = IODataset()
    dataset.add_input_variable("x2", np.array([[1.0]]))
    dataset.add_input_variable("x1", np.array([[2.0]]))
    dataset.add_output_variable("y2", np.array([[3.0]]))
    dataset.add_output_variable("y1", np.array([[4.0]]))
    print([dataset.input_names, dataset.output_names])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import IODataset
    dataset = IODataset()
    dataset.add_input_variable("x2", np.array([[1.0]]))
    dataset.add_input_variable("x1", np.array([[2.0]]))
    dataset.add_output_variable("y2", np.array([[3.0]]))
    dataset.add_output_variable("y1", np.array([[4.0]]))
    print([sorted(dataset.input_names), sorted(dataset.output_names)])
    ```

### The names of an `OptimizationDataset` keep insertion order

`OptimizationDataset.design_variable_names`, `OptimizationDataset.objective_names`, `OptimizationDataset.inequality_constraint_names`, `OptimizationDataset.equality_constraint_names` and `OptimizationDataset.observable_names` no longer return sorted values; they preserve insertion order.

After bump-gemseo, the code runs without error and without comment, but these names come in the order in which the variables were added. Check the code that depends on this order and sort the names explicitly with `sorted` where you need the GEMSEO 6.3 order, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1411](https://gitlab.com/gemseo/dev/gemseo/-/issues/1411), merge request [!2508](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2508).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.optimization_dataset import OptimizationDataset
    dataset = OptimizationDataset()
    dataset.add_design_variable("x2", np.array([[1.0]]))
    dataset.add_design_variable("x1", np.array([[2.0]]))
    dataset.add_objective_variable("f2", np.array([[3.0]]))
    dataset.add_objective_variable("f1", np.array([[4.0]]))
    dataset.add_inequality_constraint_variable("g2", np.array([[5.0]]))
    dataset.add_inequality_constraint_variable("g1", np.array([[6.0]]))
    dataset.add_equality_constraint_variable("h2", np.array([[7.0]]))
    dataset.add_equality_constraint_variable("h1", np.array([[8.0]]))
    dataset.add_observable_variable("o2", np.array([[9.0]]))
    dataset.add_observable_variable("o1", np.array([[10.0]]))
    print([
        dataset.design_variable_names, dataset.objective_names, dataset.inequality_constraint_names,
        dataset.equality_constraint_names, dataset.observable_names,
    ])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import OptimizationDataset
    dataset = OptimizationDataset()
    dataset.add_design_variable("x2", np.array([[1.0]]))
    dataset.add_design_variable("x1", np.array([[2.0]]))
    dataset.add_objective_variable("f2", np.array([[3.0]]))
    dataset.add_objective_variable("f1", np.array([[4.0]]))
    dataset.add_inequality_constraint_variable("g2", np.array([[5.0]]))
    dataset.add_inequality_constraint_variable("g1", np.array([[6.0]]))
    dataset.add_equality_constraint_variable("h2", np.array([[7.0]]))
    dataset.add_equality_constraint_variable("h1", np.array([[8.0]]))
    dataset.add_observable_variable("o2", np.array([[9.0]]))
    dataset.add_observable_variable("o1", np.array([[10.0]]))
    print([
        sorted(dataset.design_variable_names), sorted(dataset.objective_names), sorted(dataset.inequality_constraint_names),
        sorted(dataset.equality_constraint_names), sorted(dataset.observable_names),
    ])
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### `OptimizationMetadata` no longer has `use_standardized_objective`

The `use_standardized_objective` field of `OptimizationMetadata` was removed; use the `use_standardized_objective` field of the post-processor settings (via `UseStandardizedObjectiveMixin`) instead. The metadata no longer hold whether to use the standardized objective.

bump-gemseo renames the keyword `output_names_to_constraint_names` to `output_name_to_constraint_names` and marks the line with the comment `use the use_standardized_objective field of the post-processor settings instead`. If you pass `use_standardized_objective` positionally and leave it, the following positional arguments shift and the call raises `TypeError: OptimizationMetadata.__init__() got multiple values for argument 'output_name_to_constraint_names'`. Remove this argument from the call and the code that reads the attribute, as in the example below, and set this option on the post-processor settings.

!!! info "After bump-gemseo"

    - **Comment**: `use the use_standardized_objective field of the post-processor settings instead`
    - **Error with GEMSEO 7**: `TypeError: OptimizationMetadata.__init__() got multiple values for argument 'output_name_to_constraint_names'`
    - **GEMSEO 7 cannot give the GEMSEO 6.3 value here**: the metadata no longer hold whether to use the standardized objective (a post-processor setting).
    - **See** merge request [!2551](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2551).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.constraint_tolerances import ConstraintTolerances
    from gemseo.datasets.optimization_metadata import OptimizationMetadata
    metadata = OptimizationMetadata("f", "-f", False, True, ConstraintTolerances(), output_names_to_constraint_names={"g": ["g"]}, feasible_iterations=[0, 1], optimum_iteration=1)
    print([metadata.objective_name, metadata.use_standardized_objective, dict(metadata.output_names_to_constraint_names), metadata.optimum_iteration])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.dataset.optimization_metadata import OptimizationMetadata
    from gemseo.optimization.constraint_tolerances import ConstraintTolerances
    metadata = OptimizationMetadata("f", "-f", False, ConstraintTolerances(), output_name_to_constraint_names={"g": ["g"]}, feasible_iterations=[0, 1], optimum_iteration=1)
    print([metadata.objective_name, "<removed in GEMSEO 7>", dict(metadata.output_name_to_constraint_names), metadata.optimum_iteration])
    ```

### `output_names_to_constraint_names` became `output_name_to_constraint_names`

The attribute `output_names_to_constraint_names` of `OptimizationMetadata` was renamed to `output_name_to_constraint_names`.

bump-gemseo renames it only when it can infer that the object is an `OptimizationMetadata`. Otherwise, e.g. for metadata read from `misc["optimization_metadata"]` of a dataset, it marks the line with the comment `cannot transform: the type of metadata could not be inferred; if it is an instance of OptimizationMetadata, output_names_to_constraint_names became output_name_to_constraint_names`. Rename the attribute yourself, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: the type of metadata could not be inferred; if it is an instance of OptimizationMetadata, output_names_to_constraint_names became output_name_to_constraint_names`
    - **See** issue [#1797](https://gitlab.com/gemseo/dev/gemseo/-/issues/1797), merge request [!2486](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2486).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    from gemseo.problems.optimization.power_2 import Power2
    problem = Power2()
    OptimizationLibraryFactory().execute(problem, algo_name="SLSQP", max_iter=20)
    metadata = problem.to_dataset().misc["optimization_metadata"]
    print({name: list(names) for name, names in metadata.output_names_to_constraint_names.items()})
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.optimization import SLSQP_Settings
    from gemseo.optimization.factory import OptimizationLibraryFactory
    from gemseo.problem.optimization.power_2 import Power2
    problem = Power2()
    OptimizationLibraryFactory().execute(problem, settings=SLSQP_Settings(max_iter=20))
    metadata = problem.to_dataset().misc["optimization_metadata"]
    print({name: list(names) for name, names in metadata.output_name_to_constraint_names.items()})
    ```
