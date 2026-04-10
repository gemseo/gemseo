<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Chaining simulation models { #usecases-chaining-simulation-models }

## What it means { #usecases-what-it-means }

The simplest multidisciplinary use case
is to **chain** several simulation models sequentially:
the outputs of one discipline
are passed as inputs to the next one.
There is no feedback loop---data flows in one direction only.

This is the first step towards a multidisciplinary process:
connecting disciplines from different teams or tools
into a single, automated simulation pipeline.

<!-- ![A simple discipline chain](../../assets/images/user_guide/coupling_graph_chain.png){: style="display:block; margin:auto; max-width:40%" } -->

*Two disciplines chained sequentially: Disc 1 produces y, which feeds into Disc 2.*

## Why it matters { #usecases-why-it-matters }

In practice, engineers often start by running disciplines manually:
they execute a first tool,
extract its outputs,
reformat them,
and feed them into the next tool.
This manual hand-off is tedious, error-prone, and not reproducible.

By wrapping each tool as a GEMSEO discipline
and chaining them together,
the entire pipeline becomes:

- **Automated**: no manual data transfer between tools.
- **Reproducible**: the same chain produces the same results.
- **Validated**: grammars check data consistency at each step.
- **Extensible**: adding a new discipline to the chain is straightforward.

## How GEMSEO does it { #usecases-how-gemseo-does-it }

GEMSEO analyzes the inputs and outputs of each discipline
and automatically determines the execution order.
The data flow is a **consequence** of the variable naming convention:
if Disc 1 produces a variable named `y`
and Disc 2 requires an input named `y`,
GEMSEO connects them.

The resulting chain can be executed as a single discipline,
with its own inputs (the inputs not produced by any discipline in the chain)
and outputs (all the outputs produced by the disciplines).
This composite discipline can then be used
in a scenario for evaluation, DOE, or optimization.

<!-- ![XDSM of a simple optimization with a chain](../../assets/images/user_guide/xdsm_simple_optimization.png){: style="display:block; margin:auto; max-width:60%" } -->

*An XDSM diagram showing an optimizer driving a chain of two uncoupled disciplines.*

## A stepping stone { #usecases-a-stepping-stone }

Chaining disciplines is typically the **first use case**
that engineers adopt with GEMSEO.
It provides immediate benefits
(automation, validation, reproducibility)
without requiring knowledge of MDA or MDO formulations.

From there, the natural next steps are:

- Adding **coupling** when disciplines have circular dependencies (see [Coupling simulation models](coupling_simulation_models.md)).
- Adding a **DOE** to explore the design space (see [Exploring design space](exploring_design_space.md)).
- Adding an **optimizer** to find the best design (see [Optimizing a design](optimizing_a_design.md)).
