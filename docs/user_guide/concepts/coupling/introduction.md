---
description: "Discipline couplings in GEMSEO: concepts, dependency graph, MDA solving, and gradient computation."
tags: ['user_guide']
search:
  boost: 2
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Coupled systems { #concept-coupled-systems }

When dealing with multiple [Disciplines][gemseo.core.discipline.Discipline],
outputs of one discipline may connect to inputs of another.
These connections are called **couplings**.

A **weak coupling** is a one-way data flow from one discipline to another.
A **strong coupling** occurs when two disciplines form a feedback loop:
the output of one feeds into the other, whose output feeds back into the first.

!!! how-to
    - [Chain disciplines][chain-disciplines]

## The dependency graph { #concept-the-dependency-graph }

While couplings are typically defined by users,
GEMSEO creates the dependency graph automatically.
The automatic coupling detection relies on a naming convention:
two variables with the same name are treated as the same variable.
Therefore,
two disciplines sharing a variable name are automatically coupled.
This enables fast reconfiguration when modifying disciplines.

The couplings can then be visualized in different ways.
See [Coupling visualization][concept-coupling-visualization] for details.

## Solving strong couplings { #concept-solving-strong-couplings }

When dealing with strongly coupled disciplines,
Multi-Disciplinary Analyses (MDAs) are required
to ensure consistency between disciplines.
The variables involved are referred to as **coupling variables**.

See
[Solving Multi Disciplinary Analysis][concept-solving-multi-disciplinary-analysis]
for details.

## Gradient computation { #concept-coupling-gradient-computation }

The propagation of discipline gradients into the workflow is automatic in GEMSEO.
Given a set of disciplines, GEMSEO generates their connections and,
if needed,
can compute the total Jacobian matrix of the resulting chain.

See [Gradient computation][concept-coupled-gradient-computation] for details.

## Going further { #concept-going-further }

There are several ways to interfere with the GEMSEO automatic workflow generation.

### Namespaces { #concept-going-further-namespaces }

[Namespaces][concept-namespaces] can be used in the disciplines
to add a prefix to some variables,
allowing the re-use of disciplines in a single workflow.
It can be helpful when dealing with the design of multiple similar objects.

### Wrapping disciplines { #concept-going-further-wrapping-disciplines }

Variable names can be changed or variables may be redefined
by using discipline wrappers.

!!! how-to
    - [Remapping the variables][]
    - [Renaming variables][]
