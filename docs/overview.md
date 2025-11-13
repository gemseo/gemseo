<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Overview { #gemseo-overview }

GEMSEO stands for Generic Engine for Multi-disciplinary Scenarios, Exploration and Optimization.

Built on top of [NumPy](http://www.numpy.org/), [SciPy](http://scipy.org/) and [Matplotlib](http://www.matplotlib.org/) libraries, this [Python](https://www.python.org/) library enables an automatic treatment of design problems, using design of experiments, optimization algorithms and graphic post-processing.

GEMSEO is more particularly specialized in [Multidisciplinary Design Optimization](https://en.wikipedia.org/wiki/Multidisciplinary_design_optimization) (MDO).

## What is GEMSEO used for?

Let us consider a design problem to be solved over a particular design space and using different simulation software (called disciplines ).

From this information, GEMSEO carries out the following steps:

> 1. **Create** a scenario, translating this design problem into a mathematical optimization problem. The user can choose its favorite [MDO formulation](mdo/mdo_formulations.md) (or architecture) or test another one, by simply giving the name of the formulation to the scenario. [Bi-level MDO formulations](mdo/mdo_formulations.md#the-bi-level-formulation) allow to use disciplines that are themselves scenarios.
> 2. **Solve** this design problem, using either an optimization algorithm or a DOE.
> 3. **Plot** the results and logs of the optimization problem resolution.

## The power of GEMSEO

Fully based on the power of object-oriented programming, the scenario automatically generates the process, with corresponding work flow and data flow.

GEMSEO aims at pushing forward the limits of automation in simulation processes development, with a particular focus made on:

> - the integration of heterogeneous simulation environments in industrial and research contexts,
> - the integration of state-of-the-art algorithms for optimization, design of experiments, and coupled analyses,
> - the automation of MDO results analysis,
> - the development of distributed and multi-level [MDO formulations](mdo/mdo_formulations.md).

![XDSM of the bi-level formulation applied to the Sobieski's SSBJ problem](_images/bilevel_ssbj.png)
*A bi-level MDO formulation on Sobieski's SSBJ problem.*

## Main features

### Basics of MDO

- Analyse an MDO problem and generate an N2 chart and an XDSM diagram without wrapping any tool or writing code [[Read more]](interface/study_analysis.md)
- Use different optimization algorithms [[Read more]](mdo/optimization.md)
- Use different sampling methods for design of experiments  [[Read more]](mdo/optimization.md)
- Use different [MDO formulations](mdo/mdo_formulations.md): [MDF](mdo/mdo_formulations.md#the-mdf-formulation), [IDF](mdo/mdo_formulations.md#the-idf-formulation), [bilevel](mdo/mdo_formulations.md#the-bi-level-formulation) and disciplinary optimizer [[Read more]](mdo/mdo_formulations.md)
- Visualize a [MDO formulation](mdo/mdo_formulations.md) as an [XDSM diagram](mdo/mdo_formulations.md#xdsm-visualization) [[Read more]](mdo/mdo_formulations.md)
- Use different `mda` algorithms: fixed-point algorithms (Gauss-Seidel and Jacobi), root finding methods (Newton Raphson and Quasi-Newton) and hybrid techniques [[Read more]][multi-disciplinary-analyses].
- Use different surrogate models to substitute a costly discipline within a process: linear regression, RBF model and Gaussian process regression [[Read more]](surrogate.md)
- Visualize the optimization results by means of many graphs [[Read more]][how-to-deal-with-post-processing]
- Record and cache the disciplines inputs and outputs into HDF files [[Read more]][caching-and-recording-discipline-data]
- Experiment with different GEMSEO 's benchmark MDO problems [[Read more]](problems/index.md)

### Advanced techniques in MDO

- Create simple analytic disciplines using symbolic calculation [[Read more]](disciplines/analytic_discipline.md)
- Use a cheap scalable model instead of an costly discipline in order to compare different formulation performances [[Read more]](scalable_models/original_paper.md)
- Monitor the execution of a scenario using logs, [XDSM](mdo/mdo_formulations.md#xdsm-visualization) diagram or an observer design pattern [[Read more]](scenario/monitoring.md)

### Development

- Interface simulation software with GEMSEO using JSON schema based grammars for inputs and output description and a wrapping class for execution [[Read more]](interface/software_connection.md)

### Plug-in options

- Options of the available optimization algorithms [[Read more]][available-optimization-algorithms]
- Options of the available DOE algorithms [[Read more]][available-doe-algorithms]
- Options of the available MDA algorithms [[Read more]][available-mda-algorithms]
- Options of the available formulation algorithms [[Read more]][available-mdo-formulations]
- Options of the available post-processing algorithms [[Read more]][available-post-processing-algorithms]
- Options of the available machine learning algorithms [[Read more]][algorithms-of-machine-learning]
