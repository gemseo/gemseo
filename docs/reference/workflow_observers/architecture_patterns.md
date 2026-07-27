<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Workflow Observer — Architecture Patterns and Design Decisions

**For Developers and Architects** - Understanding the design rationale

## Table of Contents

- [Design Patterns Used](#design-patterns-used)
- [Key Design Decisions](#key-design-decisions)
- [Trade-offs](#trade-offs)
- [Extensibility Points](#extensibility-points)

---

## Design Patterns Used

### Observer Pattern

The core system implements a two-level observer pattern:

#### Level 1: Lifecycle Observation

- `WorkflowObserverInterface` defines observation contract
- Observers track when methods start/finish
- Non-intrusive: observed classes unaware of observers

#### Level 2: Processor Delegation

- Processor pattern separates concern of "what to observe" from "what to do"
- Observers delegate to processors for actual work
- Multiple processor types can be swapped

### Decorator Pattern

Module-level decorator functions in `injector.py` wrap existing methods:

```python
# Before decoration
def execute(self):
    return self._run()

# After decoration with _decorate_with_both
def execute(self):
    observer.start(CallSpec(...))
    try:
        result = self._run()
    finally:
        observer.end(CallSpec(...), result)
    return result
```

**Why not subclass?**

- Would require modifying user code
- Wouldn't work for built-in types
- Multiple observations harder to compose

### Metaclass Pattern

`WorkflowObserverMeta` intercepts class instantiation:

```python
class MyClass(metaclass=WorkflowObserverMeta):
    pass

# First instantiation:
# 1. Metaclass.__call__ invoked
# 2. _WorkflowObserverInjector.accept(MyClass)
# 3. If yes, _WorkflowObserverInjector.inject(MyClass)
# 4. Proceed with normal instantiation
```

**Why metaclass?**

- Single interception point
- Transparent to user code (no API changes)
- Works with inheritance and multiple instantiation

### Facade Pattern

`BaseWorkflowObserverDispatcher` routes to method-specific observers:

```python
class DisciplineWorkflowObserver(BaseWorkflowObserverDispatcher):
    _method_name_to_observer_class = {
        "execute": DisciplineExecutionWorkflowObserver,
        "linearize": DisciplineLinearizationWorkflowObserver,
    }
```

**Why facade?**

- Single entry point for multiple methods
- Routes based on method name
- Cleaner than multiple decorators

### Specification Pattern

`ObservationSpec` declaratively specifies observability:

```python
observer_spec = ObservationSpec(
    base_class="package.Class",
    excluded_sub_classes={"package.Internal"},
    method_names_for_both={"execute"},
)
```

**Why specification?**

- Declarative > imperative
- Easy to extend (add new methods)
- Single source of truth
- No method-by-method boilerplate

---

## Key Design Decisions

### Observation vs. Modification

**Question**: Should observers modify behavior or just observe?

**Answer**: Observers are read-only; processors modify system state.

**Rationale**:

- Clear separation of concerns
- Easier to reason about
- Multiple observers don't interfere
- Can disable/enable without changing code

### Transparent Injection

**Question**: Inject observers transparently or require explicit API?

**Answer**: Fully transparent via metaclass.

**Rationale**:

- Zero API changes to observed classes
- Works with legacy code
- No explicit registration needed in user code
- User enables via configuration

```python
# User code is unchanged
scenario = create_scenario(...)
scenario.execute()  # Observer automatically injected

# Only configuration needed
_configuration.directory_manager.enable = True
```

### Tree vs. Global Stack

**Question**: Track observer relationships in what structure?

**Answer**: Hierarchical tree with per-thread/process stacks.

**Rationale**:

- Nested observations are common (scenario → disciplines → MDA → iterations)
- Need to know parent for directory hierarchy
- LifoQueue provides natural nesting order
- Per-thread tracking handles multi-threading

```txt
Scenario (root)
├── Discipline 1 (child of Scenario)
├── Discipline 2 (child of Scenario)
└── MDA (child of Scenario)
    ├── Iteration 0 (child of MDA)
    └── Iteration 1 (child of MDA)

# Stack for scenario thread: [Scenario, Discipline1, ...]
# When Discipline1 ends, popped; Scenario becomes current parent
```

### Processor per Observer, Not per Method

**Question**: One processor per observer or per method?

**Answer**: One processor per observer, routes internally via dispatcher.

**Rationale**:

- Simpler factory logic
- Processor owns all tracing for its observer
- Less object creation overhead
- Consistent with lifecycle (observer is created once)

```python
# One DisciplineWorkflowObserver (dispatcher)
# Two sub-observers: DisciplineExecutionWorkflowObserver, DisciplineLinearizationWorkflowObserver
# Each has its own processor: DisciplineExecutionDMProcessor, DisciplineLinearizationDMProcessor
```

### Specification-based Matching

**Question**: How to specify observability - code or config?

**Answer**: Specification objects + code in class definition.

**Rationale**:

- Declarative > imperative
- Centralizes observability rules
- Easy to reason about (see what's observed)
- Supports inheritance naturally

```python
class DisciplineWorkflowObserver(BaseWorkflowObserverDispatcher):
    # Single place to define what's observed
    _spec = ObservationSpec(...)
```

---

## Trade-offs

### Transparency vs. Debuggability

**Trade-off**: Automatic injection hides where observers come from.

**Mitigation**:

- Clear error messages ("No observer found for class X")
- Debug info: `accept()`, `inject()` available on `_WorkflowObserverInjector`
- Documentation explains flow

### Flexibility vs. Complexity

**Trade-off**: Multiple observer types, specs, dispatchers make system complex.

**Mitigation**:

- Clear hierarchy (BaseWorkflowObserver → specialized)
- Separation of concerns (observer vs. processor)
- Built-in observers cover common cases

---

## Extensibility Points

### Add New Observer Type

**Extension Point**: Create observer class, register it

```python
# 1. src/gemseo/utils/_workflow_observers/custom.py
class CustomWorkflowObserver(BaseWorkflowObserver):
    _spec = ObservationSpec(
        base_class="mypackage.MyClass",
        method_names_for_both={"execute"},
    )

# 2. Register in src/gemseo/utils/_workflow_observers/injector.py
_WorkflowObserverInjector.register(CustomWorkflowObserver)

# 3. Create a matching processor (see directory_manager docs)
```
