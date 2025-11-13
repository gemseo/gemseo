<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Design of experiments

Design of experiments (DOE) is a branch of applied statistics to plan, conduct and analyze real or numerical experiments. It consists in selecting input values in a methodical way (sampling) and then performing the experiments to obtain output values (measurement or evaluation).

!!! note
    "DOE" may also refer to the sampling method itself, e.g. Latin hypercube sampling.

A DOE can be used to:

- determine whether an input or an interaction between inputs has an effect on an output (sensitivity analysis),
- model the relationship between inputs and outputs (surrogate modeling),
- optimize an output with respect to inputs while satisfying some constraints (trade-off).

## API { #doe-api }

In GEMSEO, a [BaseDOELibrary][gemseo.algos.doe.base_doe_library.BaseDOELibrary] contains one or several DOE algorithms.

As any [BaseDriverLibrary][gemseo.algos.base_driver_library.BaseDriverLibrary], a [BaseDOELibrary][gemseo.algos.doe.base_doe_library.BaseDOELibrary] executes an algorithm from an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] and options. Most of the DOE algorithms also need the number of samples when calling [execute()][gemseo.algos.doe.base_doe_library.BaseDOELibrary.execute]:

``` python
from gemseo.algos.doe.lib_pydoe import PyDOE
pydoe_library = PyDOE()
optimization_result = pydoe_library.execute(problem, algo_name="PYDOE_LHS", n_samples=100)
```

In the presence of an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem], it is advisable to apply DOE algorithms with the function [execute_algo()][gemseo.execute_algo] which returns an [OptimizationResult][gemseo.algos.optimization_result.OptimizationResult]:

``` python
from gemseo import execute_algo
optimization_result = execute_algo(problem, "PYDOE_LHS", algo_type="doe", n_samples=100)
```

In the presence of an [Discipline][gemseo.core.discipline.discipline.Discipline], it is advisable to create a [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario] with the function [create_scenario()][gemseo.create_scenario] and pass the DOE algorithm to [DOEScenario.execute()][gemseo.scenarios.doe_scenario.DOEScenario.execute]:

``` python
doe_scenario.execute(algo_name="PYDOE_LHS", n_samples=100)
```

## Algorithms

GEMSEO wraps different kinds of DOE algorithms from the libraries [PyDOE](https://github.com/relf/pyDOE3) and [OpenTURNS](https://openturns.github.io/www/).

!!! note
    The names of the algorithms coming from OpenTURNS starts with `"OT_"`, e.g. `"OT_OPT_LHS"`. You need to [install][installation] the full features of GEMSEO in order to use them.

!!! info "See Also"
    All the DOE algorithms and their settings are listed on [this page][available-doe-algorithms].

These DOE algorithms can be classified into categories:

- the Monte Carlo sampling generates values in the input space distributed as a multivariate uniform probability distribution with stochastically independent components; the algorithm is `"OT_MONTE_CARLO"`,
- the [low-discrepancy sequences](https://en.wikipedia.org/wiki/Low-discrepancy_sequence) are sequences of input values designed to be distributed as uniformly as possible (the deviation from uniform distribution is called *discrepancy*); the algorithms are `"OT_FAURE"`, `"OT_HALTON"`, `"OT_HASELGROVE"`, `"OT_SOBOL"` and `"OT_REVERSE_HALTON"`,
- the Latin hypercube sampling (LHS) is an algorithm generating $N$ points in the input space based on the generalization of the [Latin square](https://en.wikipedia.org/wiki/Latin_square): the range of each input is partitioned into $N$ equal intervals and, for each interval, one and only one of the points has its corresponding input value inside the interval; the algorithms are `"PYDOE_LHS"`, `"OT_LHS"` and `"OT_LHSC"`,
- the optimized LHS is an LHS optimized by Monte Carlo replicates or simulated annealing; the algorithm is `"OT_OPT_LHS"`,
- the stratified DOEs makes the inputs, also called *factors*, vary by level;
    - a full factorial DOE considers all the possible combinations of these levels across all the inputs; the algorithms are `"PYDOE_FF2N"`, `"PYDOE_FULLFACT"` and `"OT_FULLFACT"`;
    - a factorial DOE samples the diagonals of the input space, symmetrically with respect to its center; the algorithm is `"OT_FACTORIAL"`;
    - an axial DOE samples the axes of the input space, symmetrically with respect to its center; the algorithm is `"OT_AXIAL"`;
    - a central composite DOE combines a factorial and an axial DOEs; the algorithms are `"OT_COMPOSITE"` and `"PYDOE_CCDESIGN"`;
    - Box--Behnken and Plackett-Burman DOEs for response surface methodology; the algorithms are `"PYDOE_BBDESIGN"` and `"PYDOE_PBDESIGN"`.

GEMSEO also offers a [CustomDOE][gemseo.algos.doe.custom_doe.custom_doe.CustomDOE] to set its own input values, either as a CSV file or a two-dimensional NumPy array.

## Advanced use

Once the functions of the [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] have been evaluated, the input samples can be accessed with [samples][gemseo.algos.doe.base_doe_library.BaseDOELibrary.samples].

!!! note
    GEMSEO applies a DOE algorithm over a unit hypercube of the same dimension as the input space and then project the [unit_samples][gemseo.algos.doe.base_doe_library.BaseDOELibrary.unit_samples] onto the input space using either the probability distributions of the inputs, if the latter are random variables, or their lower and upper bounds.

If we do not want to evaluate the functions but only obtain the input samples, we can use the method [compute_doe()][gemseo.algos.doe.base_doe_library.BaseDOELibrary.compute_doe] which returns the samples as a two-dimensional NumPy array.

The quality of the input samples can be assessed with a [DOEQuality][gemseo.algos.doe.doe_quality.DOEQuality] computing the $\varphi_p$, minimum-distance and discrepancy criteria. The smaller these quality measures, the better, except for the minimum-distance criterion for which the larger it is the better. The qualities can be compared with logical operations, with `DOEQuality(doe_1) > DOEQuality(doe_2)` meaning that `doe_1` is better than `doe_2`.

!!! note
    When numerical metrics are not sufficient to compare two input samples sets, graphical indicators (e.g. [ScatterMatrix][gemseo.post.dataset.scatter_plot_matrix.ScatterMatrix]) could be considered.

Lastly, a [BaseDOELibrary][gemseo.algos.doe.base_doe_library.BaseDOELibrary] has a [seed][gemseo.algos.doe.base_doe_library.BaseDOELibrary.seed] initialized at 0 and each call to [execute()][gemseo.algos.doe.base_doe_library.BaseDOELibrary.execute] increments it before using it. Thus, two executions generate two distinct set of input-output samples. For the sake of reproducibility, you can pass your own seed to [execute()][gemseo.algos.doe.base_doe_library.BaseDOELibrary.execute] as a DOE option.
