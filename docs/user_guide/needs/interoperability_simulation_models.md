<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Interoperability of simulation models { #needs-interoperability-of-simulation-models }

## The problem { #needs-the-problem }

In an industrial context,
the design of a complex product involves **multiple teams**,
each responsible for a specific physics or subsystem:
aerodynamics, structures, propulsion, thermal management, etc.
Each team relies on its own **simulation software**,
written in different languages (Python, Fortran, C++, MATLAB, etc.),
using different file formats,
running on different platforms,
and following different conventions for naming variables.

Building a multidisciplinary simulation process
requires making all these heterogeneous tools work together.
Without a common framework,
this integration is a time-consuming,
error-prone and costly task
that must be redone every time a tool changes or a new one is introduced.

## GEMSEO's answer: the Discipline { #needs-gemseos-answer-the-discipline }

GEMSEO introduces the concept of **discipline**,
a unified interface that wraps any simulation software
regardless of its implementation details.
A discipline defines:

- its **inputs**: the data it needs to run,
- its **outputs**: the data it produces,
- its **execution**: how to compute outputs from inputs.

Thanks to this abstraction,
GEMSEO can orchestrate any combination of tools
without knowing their internals.
The data flow between disciplines
is determined automatically
from a consistent naming convention:
if two disciplines refer to the same variable name,
GEMSEO knows they are connected.

## Many ways to wrap simulation software { #needs-many-ways-to-wrap-simulation-software }

GEMSEO provides multiple options
to turn an existing simulation tool into a discipline:

| Source | GEMSEO wrapper |
|---|---|
| Python function | `AutoPyDiscipline` |
| Analytic expressions | `AnalyticDiscipline` (with automatic derivatives via SymPy) |
| Excel workbook | `XLSDiscipline` |
| MATLAB function | `gemseo-matlab` plugin |
| Scilab function | `gemseo-scilab` plugin |
| Executable with file I/O | `DiscFromExe` |
| Existing evaluation data (DOE) | `SurrogateDiscipline` |
| Custom Python code | Inherit from `Discipline` |

All these options can be mixed within the same process,
enabling strong flexibility and incremental adoption.
