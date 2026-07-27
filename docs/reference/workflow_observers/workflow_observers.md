<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Workflow Observer Architecture

**Target Audience:** Developers and Maintainers
**Last Updated:** April 2026

## Table of Contents

- [Workflow Observer Architecture Overview](#workflow-observer-architecture-overview)
- [Core Components](#core-components)
- [Dispatcher Pattern](#dispatcher-pattern)
- [Observer Types](#observer-types)
- [Architecture Diagrams](#architecture-diagrams)

---

## Workflow Observer Architecture Overview

GEMSEO's workflow observation system provides transparent tracking of execution lifecycle events (start/end) for GEMSEO
objects (disciplines, scenarios, MDA solvers, optimizers, DOE algorithms). Observers are automatically injected via a
metaclass and delegate actual processing to processors (
see [Directory Manager](../directory_manager/directory_manager.md)).

**Key Features:**

- **Automatic injection**: Observers are transparently injected into observable classes via a metaclass
- **Non-intrusive**: Observable classes don't need to know about observers
- **Composable**: Multiple observers can observe different aspects of execution
- **Extensible**: New observers can be registered and automatically discovered
- **Thread-safe**: Handles multi-threading and multi-processing contexts

---

## Core Components

### WorkflowObserverInterface (`interface.py`)

The abstract interface all observers must implement:

```python
class WorkflowObserverInterface:
    def __init__(object_: object, init_arguments: CallArguments) -> None: ...

    def start(call_spec: CallSpec) -> None: ...

    def end(call_spec: CallSpec, returned_data: Any) -> None: ...
```

Supporting dataclasses:

- `CallArguments`: holds `args` and `kwargs` of a call
- `CallSpec(CallArguments)`: extends with `callable_` reference

### BaseWorkflowObserver (`base_observer.py`)

Base implementation providing:

- Lifecycle management (`start()`, `end()`)
- Integration with the observer tree
- Processor delegation via `DMProcessorFactory` (module singleton `DM_PROCESSOR_FACTORY`)
- Status tracking (`Status` dataclass)

Also defined in `base_observer.py`:

- `ObservationSpec` (dataclass): Declarative specification of what to observe
- `InjectableObserver` (Protocol): Protocol for observer classes that can be injected (requires
  `_spec: ClassVar[ObservationSpec]`)

### ObservationSpec (`base_observer.py`)

Declarative specification of what to observe:

- `base_class`: Fully qualified base class name to match
- `excluded_sub_classes`: Subclasses to exclude
- `method_names_for_start`: Methods to observe start only
- `method_names_for_finish`: Methods to observe finish only
- `method_names_for_both`: Methods to observe both start and finish

### ObserverTree (`tree.py`)

Global singleton managing parent-child observer relationships:

- Maintains a stack of active observers per thread/process
- Uses `LifoQueue` for nested observations
- Thread-safe via process/thread ID tracking

### WorkflowObserverMeta (`injector.py`)

Instrumentation for classes that shall be observed:

- Metaclass that intercepts class instantiation `WorkflowObserverMeta`
- Automatically injects observers, if needed, before instantiation via `inject_observer()`

---

## Dispatcher Pattern

Some objects need different observers for different methods. `BaseWorkflowObserverDispatcher`
(`base_dispatcher.py`) implements the facade pattern, delegating to method-specific
observers based on the name of the method to observe (`_method_name_to_observer_class`):

- **DisciplineWorkflowObserver** routes `execute` → `DisciplineExecutionWorkflowObserver`, `linearize` →
  `DisciplineLinearizationWorkflowObserver`
- **MDAWorkflowObserver** routes `execute` → `MDAExecutionWorkflowObserver`, `_iterate_once` →
  `MDAIterationWorkflowObserver`

**OptimizerWorkflowObserver** uses custom `start()`/`end()` logic instead of a dispatcher, handling `execute`,
`_finalize_previous_iteration`, and `_get_early_stopping_result` methods with specialized routing.

---

## Observer Types

| Observer                                  | Base Class Observed                                         | Methods                                                                                 |
|-------------------------------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `ScenarioWorkflowObserver`                | `EvaluationScenario`                                        | `execute` (both)                                                                        |
| `DisciplineWorkflowObserver` (dispatcher) | `Discipline` (excl. `ProcessDiscipline`, `DummyDiscipline`) | `execute`, `linearize` (both)                                                           |
| `MDAWorkflowObserver` (dispatcher)        | `BaseMDASolver`                                             | `execute`, `_iterate_once` (both)                                                       |
| `OptimizerWorkflowObserver`               | `BaseOptimizationLibrary`                                   | `execute`, `_finalize_previous_iteration` (both), `_get_early_stopping_result` (finish) |
| `DOEWorkflowObserver`                     | `BaseDOELibrary`                                            | `_evaluate_functions` (both)                                                            |

---

## Architecture Diagrams

### Class Hierarchy

See `classes.puml` for the complete observer class relationships.

### Instantiation Sequence

See `injection_sequence.puml` for the metaclass injection flow.
