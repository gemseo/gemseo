<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

!!! warning
    The DirectoryManager is an experimental feature. It has been tested on toy
    problems and certain real-world applications, but it may not work as expected in
    complex workflow setups. If you find a bug, do not hesitate to
    [create an issue on Gitlab](https://gitlab.com/gemseo/dev/gemseo/-/work_items).

# Directory Manager Architecture

**Target Audience:** Developers and Maintainers
**Last Updated:** April 2026

## Table of Contents

- [DirectoryManager Overview](#directorymanager-overview)
- [Core Components](#core-components)
- [Processors](#processors)
- [Architecture Diagrams](#architecture-diagrams)

---

## DirectoryManager Overview

The directory manager creates and manages execution directories for GEMSEO workflows, organizing output by the hierarchy
of observed objects (scenarios, disciplines, MDA solvers, optimizers, DOE algorithms). It integrates with
the [workflow observer system](../workflow_observers/workflow_observers.md), which provides lifecycle events that
trigger directory creation and cleanup.

**Key Features:**

- **Hierarchical directories**: Mirrors the observer tree as nested filesystem directories
- **Homonymic handling**: Indexed suffixes (`#0`, `#1`, ...) for repeated directory names
- **Cleanup policies**: Configurable per scenario and MDA (keep all, keep last, keep solution, etc.)
- **Thread/process safety**: Thread-local CWD tracking, spawn-aware multiprocessing
- **History and residuals**: Generates OptHistoryView plots and MDA residual convergence plots

---

## Core Components

### DirectoryManager (`manager.py`)

Global object (via `BaseMultiton`) that:

- Creates execution directories with homonymic handling (indexed suffixes via `#` separator)
- Manages working directory for multi-threaded contexts (thread-local CWD tracking)
- Applies cleanup policies to remove directories
- Coordinates with processors
- Resets when the directory manager is enabled in config (it cannot be disabled once enabled)

**Key Methods:**

- `start_directory()`: Create and enter a new directory
- `end_directory()`: Exit directory and apply cleanup policy

### Settings (`settings.py`)

Pydantic configuration for the directory manager. The class is named `Settings`; it is
re-exported as `DirectoryManagerSettings` and reachable at `_configuration.directory_manager`:

- `enable`: Master switch
- `execution_root_path`: Root directory for all executions
- `clean_up_policy`: Cleanup strategy for scenarios
- `mda_clean_up_policy`: Cleanup strategy for MDA iterations
- `save_history_backup`: Save optimization history plots
- `backup_settings`: Backup configuration (file path, iteration/call triggers)
- `save_mda_residuals`: Save MDA convergence plots
- `keep_failed_executions`: Whether to keep failed executions

**Cleanup Policies:**

`clean_up_policy` uses `CleanUpPolicy` (scenarios), `mda_clean_up_policy` uses
`MDACleanUpPolicy` (MDA iterations, which supports only the first two values):

- `KEEP_ALL`: Keep all directories
- `KEEP_LAST_ONLY`: Keep only the latest
- `KEEP_SOLUTION_ONLY`: Keep only the optimal-solution directory (`CleanUpPolicy` only)
- `KEEP_BASELINE_AND_SOLUTION`: Keep the baseline and solution (`CleanUpPolicy` only)

### BaseProcessor (`_workflow_observers/base_processor.py`)

Abstract base for processors handling specific work:

- Defines processor lifecycle: `start()` and `end()`
- Abstract `observer_class` property—set as a class attribute by each concrete
  processor—naming the observer type it handles; the factory uses it for matching

### BaseDMProcessor (`processors/base.py`)

Base processor for directory management:

- Delegates filesystem operations (`start_directory()` / `end_directory()`) to the
  `DirectoryManager` singleton
- Minimal logic—mostly delegation; concrete subclasses set `observer_class` and
  override `__str__` to build the directory name

---

## Processors

| Processor | Observer Class | Directory Name |
| --- | --- | --- |
| `DisciplineExecutionDMProcessor` | `DisciplineExecutionWorkflowObserver` | `{discipline}_execution` |
| `DisciplineLinearizationDMProcessor` | `DisciplineLinearizationWorkflowObserver` | `{discipline}_linearization` |
| `MDAExecutionDMProcessor` | `MDAExecutionWorkflowObserver` | `{mda}` |
| `MDAIterationDMProcessor` | `MDAIterationWorkflowObserver` | `{mda}_iteration_{iter}` |
| `OptimizerDMProcessor` | `OptimizerWorkflowObserver` | `Optimizer_iteration_{iter+1}` |
| `ScenarioDMProcessor` | `ScenarioWorkflowObserver` | `{scenario}` |
| `DOEDMProcessor` | `DOEWorkflowObserver` | `DOE_sample_{sample_index}` |

`DMProcessorFactory` (module singleton `DM_PROCESSOR_FACTORY`) matches a processor to an
observer by testing `isinstance(observer, processor.observer_class)`. The DOE
`{sample_index}` is one-based and equals the sample's position in the DOE, so the
directory name is reproducible regardless of the (possibly parallel) evaluation order.

---

## Architecture Diagrams

### System Overview

See `architecture_overview.puml` for the full system architecture (observers + directory
manager integration), including the processor class relationships.

### Execution Flow

See `execution_with_directories.puml` for the directory creation sequence during a scenario execution.
