<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

!!! warning
    The tracer is an experimental feature, tied to the equally experimental
    [directory manager](../directory_manager/directory_manager.md). The contents
    and the layout of the trace files may change without notice. If you find a
    bug, do not hesitate to
    [create an issue on Gitlab](https://gitlab.com/gemseo/dev/gemseo/-/work_items).

# Tracer Architecture

**Target Audience:** Developers and Maintainers
**Last Updated:** August 2026

## Table of Contents

- [Tracer Overview](#tracer-overview)
- [Core Components](#core-components)
- [Tracers](#tracers)
- [Trace Data](#trace-data)
- [YAML Conversion and Dumping](#yaml-conversion-and-dumping)
- [Large Arrays](#large-arrays)
- [Failure Handling](#failure-handling)

---

## Tracer Overview

The tracer records what an observed object did during one call: the data it
consumed, the data it produced and the timing of the call. It is driven by
the [workflow observer system](../workflow_observers/workflow_observers.md),
through the processors of the
[directory manager](../directory_manager/directory_manager.md): each
processor owns one tracer and writes its trace into the execution directory
created for the observed call.

The data of an observed object is split in two:

- the **static** data, i.e. what does not change from one call to the next
  (the class documentation and the constructor arguments), written once to the
  [trace registry](#traceregistry-registrypy) when the object is first
  observed,
- the **dynamic** data, i.e. one record per observed call, written to a
  `.gemseo-trace.yml` file in the execution directory of that call, and pointing to
  the registry entry through an `object_id`.

**Key Features:**

- **One file per observed call**: `.gemseo-trace.yml`, next to the files the observed
  object itself wrote
- **Static data written once**: `<execution_root_path>/.gemseo-traces/<ClassName>/<n>.trace.yml`
- **Normalized arguments**: an argument is traced under the name of its
  parameter, whatever the way the caller passed it
- **YAML-friendly conversion**: NumPy arrays, paths, enums, settings models and
  design spaces are mapped onto types PyYAML can represent
- **Large arrays as `.npy` files**: a NumPy array of more than 16 elements is
  written to a sibling `.npy` file and referenced from the YAML, instead of
  being unwrapped element by element, see [Large Arrays](#large-arrays)

---

## Core Components

### BaseTracer (`base.py`)

Abstract base of all tracers. It holds the observee, a
[Timer][gemseo.util.timer.Timer] and the trace accumulated during one
observation cycle:

- `start(call_arguments, directory_path)`: start the timer, merge
  `_get_start_trace()` into the trace and write the arrays it referenced, if
  any, into `directory_path`
- `end(call_arguments, returned_data, directory_path)`: stop the timer, merge
  `_get_end_trace()`, write its referenced arrays into `directory_path`, then
  write `.gemseo-trace.yml` into it and reset the trace for the next cycle

Both hooks convert their data as soon as it is captured, and not when the trace
is written: a traced value is held by reference, so an observed call mutating
one of its arguments in place, e.g. a discipline adding to its input array,
would otherwise be traced with the value that argument had after the call. For
the same reason, a large array met during that conversion is written to its
`.npy` file immediately, rather than being kept until the trace itself is
written: see [Large Arrays](#large-arrays).

Subclasses do not override `start()`/`end()`; they contribute data through
three hooks:

| Hook | Called | Default contents |
| --- | --- | --- |
| `_get_init_trace(init_arguments)` | Once, at construction, for the registry payload | Class documentation, unconverted constructor arguments (`TraceRegistry.register` converts them) |
| `_get_start_trace(call_arguments)` | Before the observed call | Nothing |
| `_get_end_trace(call_arguments, returned_data)` | After the observed call | `start` timestamp and `duration` |

Every trace also carries the seed built for each cycle: the `object_id` of the
observee and its `type`.

### TraceRegistry (`registry.py`)

Global object (via `BaseMultiton`) holding the static data of each observed
object. `register()` returns an **object id** such as `MDAJacobi/0`, made of
the sanitized class name and a 0-based counter local to that class, and writes
the entry to `<root_path>/.gemseo-traces/<ClassName>/<n>.trace.yml` (`<n>` being
prefixed with the process id in a worker process, see below). The entry schema
is the `_TraceRegistryEntry` dataclass: `object_id`, `type`, `name` (omitted
when the observee has none), `documentation` and `init_arguments`. A large
constructor argument in `init_arguments` is written next to the entry, under
`<root_path>/.gemseo-traces/<ClassName>/<n>.trace.arrays/`, see
[Large Arrays](#large-arrays); `register()`, not `_get_init_trace()`, performs
that conversion, since the arrays directory is named after `<n>`, known only
once the entry's index is allocated.

Notable design points:

- **No notion of an execution root**: the registry does not depend on the
  directory manager, which calls `set_root_path` instead (see
  [Integration](#integration-with-the-directory-manager)). The root cannot be a
  constructor argument: `BaseMultiton` caches one instance per class, without
  arguments, and the registry is built by whichever of the directory manager
  and a tracer comes first. Until a root is set, no entry is written and only
  the counters advance, so that ids stay well-formed for tracers built
  directly, e.g. by unit tests.
- **One id per instance**: the id is stamped on the observee as
  `_workflow_trace_id`, so the sibling tracers of a same instance (e.g. the
  execution and linearization tracers of one discipline) share one registry
  entry.
- **Ids are unique across processes**: the lock only serializes the threads of
  one process, and the per-class counters cannot disambiguate the objects built
  in workers — a forked worker inherits a snapshot of the counters, a spawned
  one rebuilds the registry with the counters starting over while the root
  path still points to the parent's root. An object registered in a worker therefore
  gets its index prefixed with the id of that process, e.g.
  `MDAJacobi/12345-0`; the ids of the main process keep their bare index, and
  an observee unpickled from the parent carries the id stamped there.
- **The registry directory is hidden on purpose**: it sits under the execution
  root, in the very namespace in which the directory manager creates the
  execution directories, and it is created before any of them. Its name starts
  with a dot because `secure_filename` strips the leading `.` and `_` of the
  name it returns, so no observee name can sanitize to `.gemseo-traces` and hit the
  bare `mkdir` of `DirectoryManager.start_directory`.
- **Entries are written atomically**: an entry is dumped to a sibling temporary
  file, then moved into place, so a tool reading the registry while a run is
  ongoing never sees a half-written document.

### Integration with the directory manager

`BaseDMProcessor` (`_directory_manager/processor/base.py`) binds the two
systems:

- it constructs its tracer from the `_tracer_class` class attribute of the
  concrete processor,
- it constructs the `DirectoryManager` first, which injects
  `configuration.directory_manager.execution_root_path` into the registry
  with `TraceRegistry.set_root_path` before any observee is registered,
- `start()` creates the execution directory, keeps the path the manager
  returns, then starts the tracer,
- `end()` lets the tracer write `.gemseo-trace.yml` in that path, then ends the
  directory.

The registry singleton is cleared together with the `DirectoryManager`, when
the directory manager is enabled in the configuration; without that reset, the
per-class counters would keep accumulating across runs sharing a process and
ids would stop being deterministic.

---

## Tracers

| Tracer | Processor | Observed call | Trace-specific data |
| --- | --- | --- | --- |
| `_BaseDisciplineTracer` | — (base) | — | `input_data`, grammar-filtered input and output data helpers |
| `DisciplineExecutionTracer` | `DisciplineExecutionDMProcessor` | `Discipline.execute` | `input_data`, `output_data`, `cache_path` for an HDF5 cache |
| `DisciplineLinearizationTracer` | `DisciplineLinearizationDMProcessor` | `Discipline.linearize` | `input_data`, `jacobian` |
| `MDAExecutionTracer` | `MDAExecutionDMProcessor` | `BaseMDASolver.execute` | `input_data`, `output_data` |
| `MDAIterationTracer` | `MDAIterationDMProcessor` | `BaseMDASolver._iterate_once` | `iteration`, `input_data`, `output_data` |
| `OptimizerTracer` | `OptimizerDMProcessor` | Optimizer iterations | `iteration` |
| `DOETracer` | `DOEDMProcessor` | `BaseDOELibrary._evaluate_functions` | `sample_index`, `input_value` |
| `ScenarioTracer` | `ScenarioDMProcessor` | `EvaluationScenario.execute` | `objective`, `optimum` |

The discipline-based tracers filter the data with the grammars of the
observee, so that only declared inputs and outputs are traced. Where the data
comes from depends on what the observed method exposes:

- `Discipline.execute` and `Discipline.linearize` both take an `input_data`
  argument, read from the normalized call arguments; `execute` returns its
  output data, `linearize` returns the Jacobian.
- `BaseMDASolver._iterate_once` takes no argument and returns a stopping
  indicator, so the input data is the solver's merged data and the output data
  its current local output data.
- `OptimizerTracer` reads the iteration number from its observer, the owner of
  the counter, and omits it when the counter is not available yet.
- `ScenarioTracer` records the objective name and the optimum only for an
  `OptimizationProblem` with an objective, and only once a complete evaluation
  exists.

---

## Trace Data

An execution directory of a discipline contains a `.gemseo-trace.yml` such as:

```yaml
object_id: Sellar1/0
type: Sellar1
input_data:
  x_1:
  - 1.0
  x_shared:
  - 1.0
  - 0.0
  y_2:
  - 1.0
  gamma:
  - 0.2
start: 2026-08-05 09:12:33.104512
duration: 0.0012
output_data:
  y_1:
  - 0.8
```

and the matching registry entry, `.gemseo-traces/Sellar1/0.trace.yml`, is:

```yaml
object_id: Sellar1/0
type: Sellar1
name: Sellar1
documentation: |
  The discipline to compute the coupling variable $y_1$.
init_arguments:
  n: 1
  k: 1.0
```

A discipline whose `gamma` input holds more than 16 elements has it referenced
instead of unwrapped inline:

```yaml
input_data:
  gamma:
    npy: .gemseo-trace.arrays/0.npy
    dtype: float64
    shape: [10000]
```

with the array itself at `.gemseo-trace.arrays/0.npy`, next to
`.gemseo-trace.yml`.

---

## YAML Conversion and Dumping

Both writers go through `_yaml.py`:

- `convert_to_yaml_data()` maps the objects met in traced data onto types
  PyYAML can represent: an enum by its value, a NumPy array of at most 16
  elements (or of an object dtype, whatever its size) by its list, a larger
  one by a reference to a `.npy` file (see [Large Arrays](#large-arrays)), a
  NumPy scalar by its item, a `Path` by its string, a `DesignSpace` by its
  variable names, a Pydantic model field by field (plus `target_class_name`
  for a `BaseSettings`), a sequence or a set by a list; anything else falls
  back to `str()`.
- `dump_yaml()` dumps with `_TracerDumper`, which emits a multi-line string
  with the literal block style (`|`) instead of a quoted scalar mangled with
  escaped newlines. The representer is registered on that subclass only, so no
  other user of PyYAML in the process is affected. Trailing whitespace is
  stripped from every line first, since PyYAML otherwise refuses the literal
  style: a trace is therefore not byte-faithful to the traced data, a
  deliberate trade-off for readability.
- `dump_yaml_to_file()` is the writer both the tracer and the registry use: it
  dumps to a sibling temporary file and moves it onto the target path, so a
  reader sees either no file or a complete one, and a dump that raises leaves
  no file at all rather than an empty one. The move costs about 25 µs per
  file, see [Cost of the atomic write](#cost-of-the-atomic-write).

### Cost of the atomic write

Writing every trace through a temporary file adds one file system operation
per traced call, the move onto the target path. Its cost was measured on the
Sobieski MDF problem solved with SLSQP in 10 iterations, which writes 683
traces, by comparing three writers over three alternated runs, on a tmpfs and
on a btrfs file system:

| Writer                                                  | tmpfs    | btrfs    |
|---------------------------------------------------------|----------|----------|
| dump to a temporary file, then move it (the one used)   | 0.729 s  | 0.741 s  |
| dump to a string, then write the file once              | 0.719 s  | 0.740 s  |
| dump directly into the file                             | 0.711 s  | 0.727 s  |

The move costs 15 to 20 ms over the run, that is about 25 µs per trace and 2
to 3% of the traced run, which untraced takes 0.24 s. The atomic write is kept
nonetheless: writing directly into the file would let a tool reading the
traces while a run is ongoing see a half-written document, and would leave an
empty trace file behind when a dump raises; dumping to a string first avoids
the latter but not the former, for a gain within the measurement noise on
btrfs.

The literal block style is a best effort, not a guarantee. `_TracerDumper`
derives from PyYAML's C dumper whenever the installed PyYAML is built with
libyaml, which dumps about six times faster; that implementation stops its
printability analysis at U+FFFD, so a string holding an astral-plane
character, e.g. an emoji, or one of U+0085, U+2028 and U+2029, comes out as an
escaped double-quoted scalar instead. No GEMSEO source triggers this, but
user-supplied text can.

A few conversions are lossy on purpose:

- A value that contains itself, directly or through a chain of containers, is
  written as the literal string `<cycle>` at the point where the container is
  met again: it has no YAML representation, and converting it would recurse
  until the stack is exhausted. A value merely met twice as a sibling, as in
  any plain acyclic structure, is converted twice as it should be.
- A NumPy array of a kind PyYAML cannot represent is stringified: a complex
  one by casting the whole array, e.g. `(1+2j)`, an extended-precision
  floating-point one item by item, e.g. `'1.5'`, since `numpy.longdouble` has
  no Python counterpart. An object array is converted item by item, an item
  possibly holding a container of its own.
- A subclass of a scalar type is normalized to a plain value — PyYAML resolves
  a representer by the exact type — so a `pandas.Timestamp` is traced by its
  ISO 8601 string and an `int` subclass by its integer value.

Keys are not sorted: the trace keeps the insertion order of the data, which is
deterministic. A collection is dumped in the block style, one item per line
(`default_flow_style=False`).

---

## Large Arrays

A NumPy array of more than 16 elements (`_max_inline_array_size`), of a dtype
kind other than object, is written to a `.npy` file with
`numpy.save(path, value, allow_pickle=False)` instead of being converted inline.
An object-dtype array stays inline whatever its size: writing it to a `.npy`
file would need `allow_pickle=True`, which this module refuses since unpickling
executes arbitrary code.

### Choice of the threshold

Converting an array inline, element by element, costs a time proportional to
its size, while writing it to a `.npy` file costs a nearly constant time,
dominated by the creation of the file and of the arrays directory. The
threshold is set from these two costs, measured on the CPU time of one traced
call, a new directory and a new file being created at each call as a trace
does:

| Elements | Inline conversion and dump | `.npy` file, with its directory |
|---------:|---------------------------:|--------------------------------:|
|        4 |                      17 µs |                           26 µs |
|        8 |                      27 µs |                           26 µs |
|       16 |                      45 µs |                           27 µs |
|       64 |                     147 µs |                           27 µs |
|      256 |                     554 µs |                           27 µs |

The two costs break even at about 8 elements. A whole run confirms it: the
Sobieski MDF problem solved with SLSQP in 10 iterations, whose variables have
4 to 10 elements, takes 0.24 s untraced, 0.72 s traced with every array inline,
0.85 s with a threshold of 3 elements (1 409 `.npy` files and as many new
directories), and 0.72 s again with any threshold from 8 elements up. A
discipline with a 10 000-element input and output, on the other hand, goes
from about 41 ms to 0.6 ms per traced call once its arrays are written to files.

The threshold is set to 16 elements rather than to the break-even of 8, for two
reasons: the margin keeps the choice valid on a slower file system, where the
cost of a file grows while the cost of an inline conversion does not, and a
vector of up to 16 elements remains readable in the trace itself, without
opening a `.npy` file.

The arrays of a YAML file `X.yml` are stored in a sibling directory `X.arrays/`,
named `0.npy`, `1.npy`, ... in the order they are converted — so
`.gemseo-trace.yml` gets a `.gemseo-trace.arrays/` sibling, and a registry
entry `<n>.trace.yml` a `<n>.trace.arrays/` one, in the same class directory.
The directory is created only when there is at least one array to write. In
the YAML, such an array is replaced by a mapping:

```yaml
npy: <arrays directory name>/<i>.npy
dtype: float64
shape: [10000]
```

`npy` is a path relative to the directory holding the YAML file.

This is implemented by `_NpyArrayStore`, created with the name of the arrays
directory and passed as the optional `array_store` argument of
`convert_to_yaml_data()` (the default, `None`, keeps every array inline,
whatever its size). A large array handed to the store is assigned the
next index and kept by reference; `write_arrays()` then dumps every array kept
so far under a given base directory and forgets them. The index keeps
increasing across successive conversions with the same store, so that the
`start` and `end` conversions of one call cannot reuse a file name — one store
is created per call cycle, in `BaseTracer`, and one per
`TraceRegistry.register()` call, named `f"{entry_name}.trace.arrays"`.

`BaseTracer.start()` writes its arrays into the execution directory
immediately, before the observed call runs: this preserves the snapshot
semantics described above (an observed call mutating its argument in place
must not change the trace) without holding a copy of a possibly large array
around until `end()`. `end()` writes its own arrays, then the YAML trace last,
so that a reader who sees `.gemseo-trace.yml` also sees the arrays it
references, `dump_yaml_to_file()` writing that file atomically.
`TraceRegistry.register()` converts `init_arguments` outside the block
guarding the registry write, so that a conversion error still aborts the
tracer construction exactly as any other conversion error would; the arrays
are written inside that guarded block, before the entry's own YAML file.

`load_yaml_trace(file_path)` reads a trace file back with `yaml.safe_load` and
resolves such references into NumPy arrays with `numpy.load(..., allow_pickle=False)`,
by recursively replacing every mapping whose keys are exactly `npy`, `dtype`
and `shape` — the convention also means a user mapping happening to have
exactly these three keys would be read back as an array reference.

---

## Failure Handling

Tracing must never change the outcome of an observed call, nor leave the
process in a broken state:

- A parameter may simply be absent from the normalized arguments of a call —
  a variadic signature may bind it under a different name, or a subclass may
  rename it — so a tracer reads it with a default rather than indexing it,
  see `normalize_arguments_safely()` in the
  [workflow observer interface](../workflow_observers/workflow_observers.md#workflowobserverinterface-interfacepy).
- When the observed call raised, no returned data is passed to `end()`; a
  discipline tracer then reads the observee's current output data, reflecting
  whatever was produced before the failure.
- If a tracer raises an error, `BaseDMProcessor` logs it and lets the observed
  call proceed, or return, untraced. The execution directory is ended as
  usual, so the working directory is restored, and the trace is reseeded, so
  that the data of the failed cycle does not leak into the next one. An
  exception that is not an error, e.g. a `KeyboardInterrupt`, still
  propagates, with the directory ended first.
- Likewise, a registry entry that cannot be written is logged instead of
  aborting the construction of the observee. Its object id is allocated
  nonetheless, so its per-call traces stay consistent with each other; they
  merely point to a registry entry that does not exist.
- A class name that sanitizes to an empty string falls back to the raw name
  instead of raising, degrading the trace rather than aborting a construction.
- The one error left to propagate is a failure to stamp `_workflow_trace_id`
  on the observee, e.g. one defining `__slots__`: the sibling tracers of that
  observee could no longer share an id, so each would register a distinct
  entry, which is worse than failing.
