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
see [Directory Manager](../directory_manager/directory_manager.md)), which create the
execution directories and record the [execution traces](../tracer/tracer.md).

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
    def __init__(object_: object, init_arguments: StrKeyMapping) -> None: ...

    def start(call_spec: CallSpec) -> None: ...

    def end(call_spec: CallSpec, returned_data: Any) -> None: ...
```

`init_arguments` holds the normalized arguments used to instantiate the observed
object, by parameter name.

Supporting dataclass:

- `CallSpec`: holds the normalized arguments of a call, by parameter name
  (`kwargs`), and the `callable_` reference. It is built through
  `CallSpec.create_safely()`, never directly.

#### Argument normalization

The arguments of an observed call are bound to the parameters of the method
called, so that a consumer, e.g. a [tracer](../tracer/tracer.md), can read an
argument by its parameter name whatever the way the caller passed it, or did
not pass it (the omitted parameters take their default value):

- `normalize_arguments_safely(callable_, args, kwargs)`: returns the arguments
  by parameter name, binding through `inspect.Signature.bind()`, which handles
  a variadic signature too. The extra positional arguments of a `*args`
  parameter are kept as a tuple under the name of that parameter prefixed with
  `*`, e.g. `"*args"`, so that a trace shows which argument is variadic; the
  extra keyword arguments of a `**kwargs` parameter are flattened into the
  returned mapping, unless one of them has the key of another argument (a
  keyword argument named after a positional-only parameter, e.g.
  `f(self, x, /, **kwargs)` called as `f(1, x=2)`, or a key such as `"*args"`
  passed by dictionary unpacking), in which case they are left nested under
  `"**kwargs"` instead — never raised on, since Python itself accepts these
  calls. A method with variadic parameters, e.g. the constructor of
  `ConstraintAggregation`, is thus observed like any other
- `CallSpec.create_safely()` wraps that function

The signature of an observed method is computed without its first parameter,
since the decorated methods are plain functions whose signature includes
`self`, and it is cached: computing a signature is expensive relative to an
observed call, e.g. an MDA iteration.

### BaseWorkflowObserver (`base_observer.py`)

Base implementation providing:

- Lifecycle management (`start()`, `end()`)
- Integration with the observer tree
- Processor delegation via `DMProcessorFactory` (module singleton `dm_processor_factory`)
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
Although it routes by method name too, it is not a candidate for
`BaseWorkflowObserverDispatcher`: a dispatcher delegates each method to an independent child
observer with its own lifecycle, while here the observed methods drive one shared lifecycle
(`_finalize_previous_iteration` closes the observation of the current iteration when it starts
and opens the next one when it ends, sharing the status and the evaluation counter captured
by `execute`).

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
