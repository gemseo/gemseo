---
reading_time: true
description: "Overview of the reference problems and datasets bundled with GEMSEO for benchmarking and illustrating algorithms, formulations and post-processors."
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

# Introduction { #concept-benchmarking }

GEMSEO ships with a collection of reference problems and datasets.
They serve two purposes:

- *benchmarking* algorithms, formulations and post-processors,
  i.e. comparing them on problems whose behavior is well understood,
- *illustrating* GEMSEO features in tutorials, how-tos and tests.

The following pages present the available benchmarks:

- [MDO problems][concept-mdo-problems]:
  coupled multidisciplinary problems
  (Sellar, Sobieski SSBJ, aerostructure, propane combustion).
- [Scalable MDO problems][concept-scalable-mdo]:
  MDO problems whose size can be tuned to stress-test scalability.
- [Optimization problems][concept-optimization-problems]:
  analytical single-objective optimization problems.
- [Multi-objective optimization problems][concept-multiobjective-optimization-problems]:
  benchmarks with several conflicting objectives and known Pareto fronts.
- [Topology optimization][concept-topology-optimization]:
  structural topology optimization benchmarks.
- [Uncertainty problems][concept-uncertainty-problems]:
  problems for uncertainty quantification and sensitivity analysis.
- [ODE problems][concept-ode-problems]:
  ordinary differential equation benchmarks.
- [Datasets][concept-benchmarking-datasets]:
  reference datasets for machine learning and data analysis.

!!! seealso

    For a complete framework to benchmark optimization algorithms
    (running scenarios, collecting performance histories and generating reports),
    see the [gemseo-benchmark](https://gitlab.com/gemseo/dev/gemseo-benchmark) plugin.
