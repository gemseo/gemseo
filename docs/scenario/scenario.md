---
status: draft
description: ""
tags: ['user_guide']
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

<!--
Contributors:
         :author: Matthias De Lozzo
-->

# Scenario { #scenario-user-guide}

## How is a scenario defined?

### What is a scenario?

A scenario is an interface that:

- creates an optimization or sampling problem,
- from a set of disciplines and a multidisciplinary formulation based on a design space and on an objective name,
- executes it from an optimization or sampling algorithm with mandatory arguments and options and
- post-process it.

### How does a scenario is implemented in GEMSEO?

Programmatically speaking, scenarios are implemented in GEMSEO through the [MDOScenario][gemseo.scenarios.mdo.MDOScenario] class.
They can be executed using
either optimizers in the case of optimization processes
or DOE algorithms in the case of trade-off studies and sampling processes.

An [MDOScenario][gemseo.scenarios.mdo.MDOScenario] is defined by four main elements:

- the `disciplines` attribute: the list of [Discipline][gemseo.core.discipline.discipline.Discipline],
- the `formulation` attribute: the multidisciplinary formulation based on [DesignSpace][gemseo.algos.design_space.DesignSpace],
- the `optimization_result` attribute: the optimization results,
- the `post_factory` attribute: the post-processing set of methods.
