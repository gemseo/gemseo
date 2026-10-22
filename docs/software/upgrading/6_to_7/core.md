---
description: "Migrating core from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Core

The changes of GEMSEO 7 about grammars, caches, namespaces, derivatives, execution and monitoring that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Silent changes

The code runs after bump-gemseo, without error nor comment, and gives other results.

### `split_namespace` splits on the last separator only

`split_namespace` now splits a name into the `(namespace, name)` pair on the last separator only, as documented; it previously split on every separator, returning a list of all segments.
bump-gemseo changes nothing here: the code runs without error after the migration, but gives other results for a name with several namespaces, e.g. `outer:inner:x`.

Check each use of `split_namespace`: keep it if you want the namespace and the name, and if you relied on all the segments, split the name yourself with the `split` method of the string, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1269](https://gitlab.com/gemseo/dev/gemseo/-/issues/1269), merge request [!2529](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2529).

=== "GEMSEO 6.3"

    ```python
    from gemseo.core.namespaces import split_namespace
    print([list(split_namespace("outer:inner:x")), list(split_namespace("x"))])
    ```

=== "GEMSEO 7"

    ```python
    print(["outer:inner:x".split(":"), "x".split(":")])
    ```

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### Caches are dictionaries whose keys are the inputs

In GEMSEO 6.3, iterating over a cache yielded `CacheEntry` objects, named tuples of the inputs, the outputs and the Jacobian.
In GEMSEO 7, caches are dictionaries whose keys are the inputs and values are the cache entries: `BaseCache.__iter__` yields the input data, the `CacheEntry.inputs` field was removed and a cache entry has only the outputs and the Jacobian, so unpacking three fields fails.

bump-gemseo rewrites the iteration over the entries into an iteration over `values()`, but leaves no comment where the inputs are read, which raises `AttributeError: 'CacheEntry' object has no attribute 'inputs'`.
Iterate over the cache to get the input data, over `cache.items()` to get the input data with their entry, and unpack `cache[key]` into `outputs, jacobian`, the inputs being the key itself, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'CacheEntry' object has no attribute 'inputs'`
    - **See** merge request [!2551](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2551).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.caches.memory_full_cache import MemoryFullCache
    cache = MemoryFullCache()
    cache.cache_outputs({"x": np.array([1.0])}, {"y": np.array([2.0])})
    cache.cache_outputs({"x": np.array([3.0])}, {"y": np.array([6.0])})
    inputs_seen = [entry.inputs["x"] for entry in cache]
    outputs_seen = [entry.outputs["y"] for entry in cache]
    inputs, outputs, jacobian = cache[{"x": np.array([3.0])}]
    print([inputs_seen, outputs_seen, inputs["x"], outputs["y"]])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.core.cache.memory_full import MemoryFullCache
    cache = MemoryFullCache()
    cache.cache_outputs({"x": np.array([1.0])}, {"y": np.array([2.0])})
    cache.cache_outputs({"x": np.array([3.0])}, {"y": np.array([6.0])})
    inputs_seen = [input_data["x"] for input_data in cache]
    outputs_seen = [entry.outputs["y"] for _, entry in cache.items()]
    key = {"x": np.array([3.0])}
    outputs, jacobian = cache[key]
    print([inputs_seen, outputs_seen, key["x"], outputs["y"]])
    ```

### `traverse_add_diff_io` is replaced by `set_differentiated_ios`

`gemseo.core.derivatives.chain_rule` was removed and its function `traverse_add_diff_io` is replaced by `set_differentiated_ios` of `gemseo.core.derivative.graph_traversal`, which has no `add_differentiated_ios` argument.
bump-gemseo renames the function but keeps its arguments without a comment, so the call raises `TypeError: set_differentiated_ios() got an unexpected keyword argument 'add_differentiated_ios'`.
Remove the `add_differentiated_ios` argument, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: set_differentiated_ios() got an unexpected keyword argument 'add_differentiated_ios'`
    - **See** merge request [!2518](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2518).

=== "GEMSEO 6.3"

    ```python
    from gemseo.core.dependency_graph import DependencyGraph
    from gemseo.core.derivatives.chain_rule import traverse_add_diff_io
    from gemseo.disciplines.analytic import AnalyticDiscipline
    first = AnalyticDiscipline({"y": "2*x"}, name="A")
    second = AnalyticDiscipline({"z": "y + w"}, name="B")
    graph = DependencyGraph([first, second]).graph
    traverse_add_diff_io(graph, ["x"], ["z"], add_differentiated_ios=True)
    print([sorted(first._differentiated_input_names), sorted(second._differentiated_output_names)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.core.dependency_graph import DependencyGraph
    from gemseo.core.derivative.graph_traversal import set_differentiated_ios
    from gemseo.discipline import AnalyticDiscipline
    first = AnalyticDiscipline({"y": "2*x"}, name="A")
    second = AnalyticDiscipline({"z": "y + w"}, name="B")
    graph = DependencyGraph([first, second]).graph
    set_differentiated_ios(graph, ["x"], ["z"])
    print([sorted(first._differentiated_input_names), sorted(second._differentiated_output_names)])
    ```

### Execution sequences are enabled through `is_enabled`

The `enable()` and `disable()` methods of `BaseExecutionSequence` were removed; the state of an execution sequence is set through its `is_enabled` attribute.
bump-gemseo leaves the calls without a comment and they raise `AttributeError: 'ExecutionSequence' object has no attribute 'disable'`.
Replace `sequence.disable()` by `sequence.is_enabled = False` and `sequence.enable()` by `sequence.is_enabled = True`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'ExecutionSequence' object has no attribute 'disable'`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2309](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2309).

=== "GEMSEO 6.3"

    ```python
    from gemseo.core._process_flow.execution_sequences.execution_sequence import ExecutionSequence
    from gemseo.disciplines.analytic import AnalyticDiscipline
    sequence = ExecutionSequence(AnalyticDiscipline({"y": "x"}))
    sequence.disable()
    disabled = sequence.is_enabled
    sequence.enable()
    print([disabled, sequence.is_enabled])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.core._process_flow.execution_sequence.execution_sequence import ExecutionSequence
    from gemseo.discipline import AnalyticDiscipline
    sequence = ExecutionSequence(AnalyticDiscipline({"y": "x"}))
    sequence.is_enabled = False
    disabled = sequence.is_enabled
    sequence.is_enabled = True
    print([disabled, sequence.is_enabled])
    ```

### Grammars reject unknown keyword arguments

`JSONGrammar.__init__`, `SimpleGrammar.__init__` and `PydanticGrammar.__init__` no longer accept and silently discard arbitrary keyword arguments.
bump-gemseo leaves the call without a comment and it raises a `TypeError`, e.g. `TypeError: SimpleGrammar.__init__() got an unexpected keyword argument 'unused_option'`.

Remove the keyword arguments that the grammar does not define, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: SimpleGrammar.__init__() got an unexpected keyword argument 'unused_option'`
    - **See** merge request [!2601](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2601).

=== "GEMSEO 6.3"

    ```python
    from gemseo.core.grammars.json_grammar import JSONGrammar
    from gemseo.core.grammars.simple_grammar import SimpleGrammar
    simple = SimpleGrammar("s", names_to_types={"a": float}, unused_option=True)
    json_grammar = JSONGrammar("j", unused_option=True)
    print([sorted(simple), sorted(json_grammar)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.core.grammar.json import JSONGrammar
    from gemseo.core.grammar.simple import SimpleGrammar
    simple = SimpleGrammar("s", name_to_type={"a": float})
    json_grammar = JSONGrammar("j")
    print([sorted(simple), sorted(json_grammar)])
    ```

### `BaseGrammar.update` refuses to nest namespaces by default

In GEMSEO 6.3, `BaseGrammar.update` merged a grammar whose element is mapped to a namespaced name into a grammar where the same element is mapped to another namespaced name.
In GEMSEO 7, it raises a `ValueError` when an element name would map to two different namespaced names, unless `allow_namespace_nesting=True`, which only the process disciplines pass.

bump-gemseo leaves the call without a comment and it raises `ValueError: The name 'x' is already mapped to the namespaced name 'a:x'; mapping it to 'b:x' would require nesting, which is only allowed for process disciplines.`
Check that this merge is intended; if it is, pass `allow_namespace_nesting=True` to `update`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ValueError: The name 'x' is already mapped to the namespaced name 'a:x'; mapping it to 'b:x' would require nesting, which is only allowed for process disciplines.`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.core.grammars.simple_grammar import SimpleGrammar
    first = SimpleGrammar("first")
    first.update_from_names(["x"])
    first.add_namespace("x", "a")
    second = SimpleGrammar("second")
    second.update_from_names(["x"])
    second.add_namespace("x", "b")
    first.update(second)
    print(sorted(first))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.core.grammar.simple import SimpleGrammar
    first = SimpleGrammar("first")
    first.update_from_names(["x"])
    first.add_namespace("x", "a")
    second = SimpleGrammar("second")
    second.update_from_names(["x"])
    second.add_namespace("x", "b")
    first.update(second, allow_namespace_nesting=True)
    print(sorted(first))
    ```

### `Discipline.LinearizationMode` no longer has `DIRECT`, `ADJOINT` and `REVERSE`

`Discipline.LinearizationMode` no longer exposes the `DerivationMode` members `DIRECT`, `ADJOINT` and `REVERSE`: a discipline either computes its analytical Jacobian (`AUTO`) or approximates it, so the enumeration now holds only `AUTO`, the approximation modes and the hybrid ones.
bump-gemseo leaves these members without a comment and they raise `AttributeError: type object 'LinearizationMode' has no attribute 'ADJOINT'`.
Use the members of `DerivationMode`, imported from `gemseo.enum`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: type object 'LinearizationMode' has no attribute 'ADJOINT'`
    - **See** issue [#1812](https://gitlab.com/gemseo/dev/gemseo/-/issues/1812), merge request [!2533](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2533).

=== "GEMSEO 6.3"

    ```python
    from gemseo.core.discipline.discipline import Discipline
    print([Discipline.LinearizationMode.ADJOINT, Discipline.LinearizationMode.DIRECT, Discipline.LinearizationMode.REVERSE])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.enum import DerivationMode
    print([DerivationMode.ADJOINT, DerivationMode.DIRECT, DerivationMode.REVERSE])
    ```

### `Monitoring` takes an execution sequence instead of a scenario

The `scenario` argument of `Monitoring` was replaced by `execution_sequence` of type `BaseExecutionSequence`, and the `workflow` attribute of `Monitoring` was removed.
bump-gemseo leaves the call without a comment and passing a scenario raises `AttributeError: 'MDOScenario' object has no attribute 'set_observer'`.

Pass `scenario.get_process_flow().get_execution_flow()` to `Monitoring`, and use this execution sequence where you used `workflow`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'MDOScenario' object has no attribute 'set_observer'`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2314](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2314).

=== "GEMSEO 6.3"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.monitoring import Monitoring
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.scenarios.mdo_scenario import MDOScenario
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=1.0, value=0.9)
    scenario = MDOScenario([AnalyticDiscipline({"y": "(a - 0.25)**2"})], "y", space, formulation_name="DisciplinaryOpt")
    monitoring = Monitoring(scenario)
    print([type(monitoring.workflow).__name__.endswith("ExecSequence") or type(monitoring.workflow).__name__.endswith("Sequence")])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.core.monitoring import Monitoring
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.formulation import DisciplinaryOpt_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.space import DesignSpace
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=1.0, value=0.9)
    scenario = MDOScenario([AnalyticDiscipline({"y": "(a - 0.25)**2"})], space, formulation_settings=DisciplinaryOpt_Settings())
    scenario.add_objective("y")
    sequence = scenario.get_process_flow().get_execution_flow()
    monitoring = Monitoring(sequence)
    print([type(sequence).__name__.endswith("Sequence")])
    ```

### `PydanticGrammar.set_descriptions` was removed

The `set_descriptions` method of `PydanticGrammar` was removed because it was deprecated.
bump-gemseo leaves the call without a comment and it raises `AttributeError: 'PydanticGrammar' object has no attribute 'set_descriptions'`.
Update the `descriptions` mapping of the grammar instead, with `grammar.descriptions.update`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'PydanticGrammar' object has no attribute 'set_descriptions'`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2309](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2309).

=== "GEMSEO 6.3"

    ```python
    from pydantic import BaseModel
    from gemseo.core.grammars.pydantic_grammar import PydanticGrammar


    class Model(BaseModel):
        a: float = 1.0


    grammar = PydanticGrammar("g", model=Model)
    grammar.set_descriptions({"a": "The variable a."})
    print(dict(grammar.descriptions))
    ```

=== "GEMSEO 7"

    ```python
    from pydantic import BaseModel
    from gemseo.core.grammar.pydantic import PydanticGrammar


    class Model(BaseModel):
        a: float = 1.0


    grammar = PydanticGrammar("g", model=Model)
    grammar.descriptions.update({"a": "The variable a."})
    print(dict(grammar.descriptions))
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### `NameMapping` takes an input mapping and an output mapping

The constructor of the `NameMapping` data processor now requires two mappings, `input_mapping` and `output_mapping`, instead of a single one, and exposes them through the `input_mapping` and `output_mapping` properties; the `mapping` and `reverse_mapping` attributes were removed.
bump-gemseo marks the call with the comment "split the mapping into input_mapping and output_mapping, two positional arguments"; left as is, the call raises `TypeError: NameMapping.__init__() missing 1 required positional argument: 'output_mapping'`.

To keep the GEMSEO 6.3 behaviour, pass the same mapping twice, `NameMapping(mapping, mapping)`, read `input_mapping` instead of `mapping`, and compute the inverse of `output_mapping` where you used `reverse_mapping`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `split the mapping into input_mapping and output_mapping, two positional arguments`
    - **Error with GEMSEO 7**: `TypeError: NameMapping.__init__() missing 1 required positional argument: 'output_mapping'`
    - **See** issue [#1659](https://gitlab.com/gemseo/dev/gemseo/-/issues/1659), merge request [!2609](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2609); issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2309](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2309).

=== "GEMSEO 6.3"

    ```python
    from gemseo.core.discipline.data_processor import NameMapping
    processor = NameMapping({"global_x": "x", "global_y": "y"})
    local = processor.pre_process_data({"global_x": 1.0})
    global_data = processor.post_process_data({"y": 2.0})
    print([local, global_data, dict(processor.mapping), dict(processor.reverse_mapping)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.core.discipline.data_processor import NameMapping
    mapping = {"global_x": "x", "global_y": "y"}
    processor = NameMapping(mapping, mapping)
    local = processor.pre_process_data({"global_x": 1.0})
    global_data = processor.post_process_data({"y": 2.0})
    print([local, global_data, dict(processor.input_mapping), {value: key for key, value in processor.output_mapping.items()}])
    ```

## Deprecated names

The code still works with GEMSEO 7, with a deprecation warning.

### Renamed attributes of objects created by `BaseFactory.create`

GEMSEO 7 renames several attributes, e.g. `output_names_to_symbols` of `AnalyticDiscipline` is now `output_name_to_symbols`, `BaseCache.last_entry` is now `BaseCache.last_item`, `Dataset.DEFAULT_GROUP` is now `Dataset.default_group` and the member `DatasetClassName.Dataset` is now `DATASET`.
bump-gemseo renames these attributes on the objects returned by `create_cache`, `create_dataset`, `create_discipline` and `DisciplineFactory.create`, but not on an object created by calling `BaseFactory.create` on its class, as in `BaseFactory.create(DisciplineFactory(), ...)`, where `output_names_to_symbols` is left unchanged.

The code still runs, but uses a deprecated name, as most renames of GEMSEO 7 keep working with a `DeprecationWarning` for one deprecation cycle.
Rename the remaining attributes by hand, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: it works, with a deprecation warning.

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo import create_cache
    from gemseo import create_dataset
    from gemseo import create_discipline
    from gemseo.core.base_factory import BaseFactory
    from gemseo.datasets import DatasetClassName
    from gemseo.disciplines.factory import DisciplineFactory
    cache = create_cache("SimpleCache")
    cache.cache_outputs({"x": np.array([1.0])}, {"y": np.array([2.0])})
    dataset = create_dataset("d", np.array([[1.0, 2.0]]), ["a", "b"], class_name=DatasetClassName.Dataset)
    discipline = create_discipline("AnalyticDiscipline", expressions={"y": "2*x"})
    other = DisciplineFactory().create("AnalyticDiscipline", expressions={"z": "x + w"})
    third = BaseFactory.create(DisciplineFactory(), "AnalyticDiscipline", expressions={"t": "x"})
    print([
        cache.last_entry.outputs["y"], dataset.DEFAULT_GROUP, sorted(discipline.output_names_to_symbols),
        sorted(other.output_names_to_symbols["z"]), sorted(third.output_names_to_symbols),
    ])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo import create_cache
    from gemseo import create_dataset
    from gemseo import create_discipline
    from gemseo.core.base_factory import BaseFactory
    from gemseo.discipline.factory import DisciplineFactory
    from gemseo.enum import DatasetClassName
    cache = create_cache("SimpleCache")
    cache.cache_outputs({"x": np.array([1.0])}, {"y": np.array([2.0])})
    dataset = create_dataset("d", np.array([[1.0, 2.0]]), ["a", "b"], class_name=DatasetClassName.DATASET)
    discipline = create_discipline("AnalyticDiscipline", expressions={"y": "2*x"})
    other = DisciplineFactory().create("AnalyticDiscipline", expressions={"z": "x + w"})
    third = BaseFactory.create(DisciplineFactory(), "AnalyticDiscipline", expressions={"t": "x"})
    print([
        cache.last_item.outputs["y"], dataset.default_group, sorted(discipline.output_name_to_symbols),
        sorted(other.output_name_to_symbols["z"]), sorted(third.output_name_to_symbols),
    ])
    ```

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. Unless an entry above gives an equivalent, they have no replacement: remove the code that uses them.

- `DisciplineJacApprox.generator_class`
- `MDOLinearFunction.normalize`
- `gemseo.core.derivatives.jacobian_operator.Operand1T`
- `gemseo.core.derivatives.jacobian_operator.Operand2T`

??? note "Private names (19), used by some plugins"

    - `gemseo.algos.base_algo_factory._AlgoFactoryMeta`
    - `gemseo.core._discipline_class_injector`
    - `gemseo.core.derivatives.chain_rule._apply_diff_ios`
    - `gemseo.core.derivatives.chain_rule._bfs_one_way_diff_io`
    - `gemseo.core.derivatives.chain_rule._initialize_add_diff_io`
    - `gemseo.core.derivatives.chain_rule._merge_diff_io_special`
    - `gemseo.core.derivatives.chain_rule._merge_diff_ios`
    - `gemseo.core.derivatives.jacobian_operator._BaseComposedOperation`
    - `gemseo.core.derivatives.jacobian_operator._ComposedOperationArrayOperator`
    - `gemseo.core.derivatives.jacobian_operator._ComposedOperationOperatorArray`
    - `gemseo.core.derivatives.jacobian_operator._ComposedOperationOperatorOperator`
    - `gemseo.core.derivatives.jacobian_operator._OperandT`
    - `gemseo.core.derivatives.jacobian_operator._SubOperationWithArray`
    - `gemseo.core.derivatives.jacobian_operator._SumOperationWithArray`
    - `gemseo.core.derivatives.mda_derivatives._replace_strongly_coupled`
    - `gemseo.core.parallel_execution.callable_parallel_execution._QueueInType`
    - `gemseo.core.parallel_execution.callable_parallel_execution._QueueOutItem2`
    - `gemseo.core.parallel_execution.callable_parallel_execution._QueueOutType`
    - `gemseo.core.parallel_execution.callable_parallel_execution._execute_workers`
