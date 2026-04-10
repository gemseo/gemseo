<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Data validation and persistence { #needs-data-validation-and-persistence }

## The problem { #needs-the-problem }

A multidisciplinary simulation process
exchanges large amounts of data between disciplines.
A wrong data type, a missing variable,
or an array of the wrong size
can cause a discipline to fail---sometimes silently,
producing incorrect results
that are only discovered much later.

In addition,
each discipline evaluation can be **expensive**
(minutes, hours, or even days of computation).
Losing the results of a crashed process,
or re-evaluating a discipline unnecessarily at the same input,
wastes precious computing resources.

## GEMSEO's answer { #needs-gemseos-answer }

### Data validation with grammars { #needs-data-validation-with-grammars }

Every discipline in GEMSEO declares its inputs and outputs
through **grammars**: formal descriptions
of the expected variable names, types, and structures.
GEMSEO validates the data **before** executing a discipline
and **after** it produces outputs.

If a required input is missing,
has the wrong type, or violates the schema,
GEMSEO raises a clear error message
identifying **all** issues in a single pass---not just the first one.
This avoids the painful cycle
of running a costly process,
discovering one error, fixing it, and running again.

Grammars are based on **JSON Schema**,
a widely adopted standard supported by many languages and tools.
This makes it straightforward
to interface GEMSEO with external technologies:
one only needs to translate the external tool's I/O description
into a JSON Schema.

### Data persistence with caches { #needs-data-persistence-with-caches }

GEMSEO provides a built-in **caching mechanism**
to store the evaluations (inputs, outputs, and Jacobians) of each discipline.

The cache serves multiple purposes:

- **Avoid redundant evaluations**: before executing a discipline, GEMSEO checks whether it has already been evaluated at the same input. If so, the cached result is returned immediately.
- **Post-processing**: cached data can be used for visualization, statistics, machine learning, or debugging after the process has completed.
- **Crash recovery**: cached data on disk allows restarting a process from where it left off.

GEMSEO offers several cache types:

| Cache type | Storage | Description |
|---|---|---|
| `SimpleCache` | Memory | Stores only the last evaluation (default). |
| `MemoryFullCache` | Memory | Stores all evaluations in memory. |
| `HDF5Cache` | Disk | Stores all evaluations in an HDF5 file, a standard format for scientific data. |

The HDF5 format is particularly well suited for engineering data:
it supports hierarchical organization,
large datasets, and can be inspected
with tools like HDFView.
Multiple disciplines can share the same HDF5 file,
each writing to its own group.
