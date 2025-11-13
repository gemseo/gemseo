<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Contributors:
         :author: Damien Guénot, Francois Gallard
-->

# How to deal with post-processing

In this section we describe the post processing features of GEMSEO, used to
analyze [OptimizationResult][gemseo.algos.optimization_result.OptimizationResult], called the
optimization history.

## What data to post-process?

Post-processing features are applicable to any
[OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] that has been solved,
which may have been loaded from the disk.

In practice,

- a [BaseScenario][gemseo.scenarios.base_scenario.BaseScenario] instance has an [BaseMDOFormulation][gemseo.formulations.base_mdo_formulation.BaseMDOFormulation] attribute,
- a [BaseMDOFormulation][gemseo.formulations.base_mdo_formulation.BaseMDOFormulation] instance has an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] attribute,
- an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] instance has an [OptimizationResult][gemseo.algos.optimization_result.OptimizationResult] attribute.

## Illustration on the Sobieski use case

The post-processing features are illustrated on MDO results obtained on the [Sobieski's SSBH problem][sobieskis-ssbj-test-case],
using different types of `formulation` ([MDF][the-mdf-formulation], [IDF][the-idf-formulation], ...)

The following code sets up and executes the problem. It is possible to try different types of MDO strategies by changing
the `formulation` value. For a detailed explanation on how to setup the case,
please see [this example][application-sobieskis-super-sonic-business-jet-mdo].

``` python
from gemseo import create_discipline
from gemseo import create_scenario

disciplines = create_discipline(
    ["SobieskiPropulsion", "SobieskiAerodynamics", "SobieskiMission", "SobieskiStructure"]
)

scenario = create_scenario(
    disciplines,
    "y_4",
    "design_space.csv",
    formulation_name="MDF",
    maximize_objective=True
)

scenario.set_differentiation_method("user")

for constraint in ["g_1","g_2","g_3"]:
      scenario.add_constraint(constraint, 'ineq')

scenario.execute(algo_name="SLSQP", max_iter=10)
```

### How to apply a post-process feature?

From this `scenario`, we can apply any kind of post-processing dedicated to [BaseScenario][gemseo.scenarios.base_scenario.BaseScenario] instances,
using either the [post_process()][gemseo.scenarios.base_scenario.BaseScenario.post_process] method or the [execute_post()][gemseo.execute_post] high-level function.

!!! note
      Only design variables and functions (objective function, constraints) are stored for post-processing.
      If you want to be able to plot state variables, you must add them as observables before the problem is executed.
      Use the [add_observable()][gemseo.scenarios.base_scenario.BaseScenario.add_observable] method.
