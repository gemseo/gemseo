<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Contributors:
         :author: Francois Gallard
-->

# Optimization and DOE framework

In this section we describe GEMSEO's optimization and DOE framework.

The standard way to use GEMSEO is through an [MDOScenario][gemseo.scenarios.mdo_scenario.MDOScenario], which
automatically creates an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] from an [MDO formulation][mdo-formulations] and a set of
[Discipline][gemseo.core.discipline.discipline.Discipline].

However, one may be interested in directly creating an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] using the class [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem],
which can be solved using an optimization algorithm or sampled with a DOE algorithm.

!!! warning
      [MDO formulation][mdo-formulations] and optimization problem developers should also understand this part of GEMSEO.

## Setting up an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]

The [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] class is composed of at least a
[DesignSpace][gemseo.algos.design_space.DesignSpace] created from [create_design_space()][gemseo.create_design_space] which describes the design variables:

``` python
from gemseo import create_design_space
from numpy import ones

design_space = create_design_space()
design_space.add_variable("x", 1, lower_bound=-2., upper_bound=2.,
                        value=-0.5 * ones(1))
```

and an objective function, of type [MDOFunction][gemseo.core.mdo_functions.mdo_function.MDOFunction]. The [MDOFunction][gemseo.core.mdo_functions.mdo_function.MDOFunction] is callable and requires at least
a function pointer to be instantiated. It supports expressions and the +, -, \ * operators:

``` python
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from numpy import cos
from numpy import exp
from numpy import sin

f_1 = MDOFunction(sin, name="f_1", jac=cos, expr="sin(x)")
f_2 = MDOFunction(exp, name="f_2", jac=exp, expr="exp(x)")
f_1_sub_f_2 = f_1 - f_2
```

From this [DesignSpace][gemseo.algos.design_space.DesignSpace],
an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem] is built:

``` python
from gemseo.algos.optimization_problem import OptimizationProblem
problem = OptimizationProblem(design_space)
```

To set the objective [MDOFunction][gemseo.core.mdo_functions.mdo_function.MDOFunction],
the attribute [objective][gemseo.algos.optimization_problem.OptimizationProblem.objective] of the [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
must be set with the objective function pointer:

``` python
problem.objective = f_1_sub_f_2
```

Similarly the [constraints][gemseo.algos.optimization_problem.OptimizationProblem.constraints] attribute must be set with a list of inequality or equality constraints.
The [f_type][gemseo.core.mdo_functions.mdo_function.MDOFunction.f_type] attribute of [MDOFunction][gemseo.core.mdo_functions.mdo_function.MDOFunction] shall be set to `"eq"` or `"ineq"` to declare the type of constraint to equality or inequality.

!!! warning
      **All inequality constraints must be negative by convention**, whatever the optimization algorithm used to solve the problem.

## Solving the problem by optimization

Once the optimization problem created, it can be solved using one of the available
optimization algorithms from the [OptimizationLibraryFactory][gemseo.algos.opt.factory.OptimizationLibraryFactory],
by means of the method [BaseOptimizationLibrary.execute()][gemseo.algos.opt.base_optimization_library.BaseOptimizationLibrary.execute]
whose mandatory arguments are the [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
and the optimization algorithm name. For example, in the case of the [L-BFGS-B algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
with normalized design space, we have:

``` python
from gemseo.algos.opt.factory import OptimizationLibraryFactory
opt = OptimizationLibraryFactory().execute(problem, algo_name="L-BFGS-B",
                                 normalize_design_space=True)
print(f"Optimum = {opt.f_opt}")
```

Note that the [L-BFGS-B algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS) is implemented in the external
library [SciPy](https://scipy.org/)
and interfaced with GEMSEO through the class [ScipyOpt][gemseo.algos.opt.scipy_local.scipy_local.ScipyOpt].

The list of available algorithms depend on the local setup of GEMSEO, and the installed
optimization libraries. It can be obtained using :

``` python
algo_list = OptimizationLibraryFactory().algorithms
print(f"Available algorithms: {algo_list}")
```

The optimization history can be saved to the disk for further analysis,
without having to re-execute the optimization.
For that, we use the method [to_hdf()][gemseo.algos.optimization_problem.OptimizationProblem.to_hdf]:

``` python
problem.to_hdf("simple_opt.hdf5")
```

## Solving the problem by DOE

DOE algorithms can also be used to sample the design space and observe the
value of the objective and constraints

``` python

from gemseo.algos.doe.factory import DOELibraryFactory

# And solve it with GEMSEO interface
opt = DOELibraryFactory().execute(
   problem, algo_name="PYDOE_LHS", n_samples=10, normalize_design_space=True
)
```

## Results analysis

The optimization history can be plotted using one of the post-processing tools,
see [this page][how-to-deal-with-post-processing].

``` python
from gemseo import execute_post

execute_post(problem, post_name="OptHistoryView", save=True, file_path="simple_opt")

# Also works from disk
execute_post("my_optim.hdf5", post_name="OptHistoryView", save=True, file_path="opt_view_from_disk")
```

![Objective function history for the simple analytic optimization](../_images/doe/simple_opt.png)

## DOE algorithms

GEMSEO is interfaced with two packages that provide DOE algorithms:
[pyDOE](https://pythonhosted.org/pyDOE/), and
[OpenTURNS](https://openturns.github.io/www/).
To list the available DOE algorithms in the current GEMSEO configuration, use
[get_available_doe_algorithms()][gemseo.get_available_doe_algorithms].

The set of plots below shows plots using various available algorithms.

- Full factorial DOE from pyDOE
![Full factorial DOE from pyDOE](../_images/doe/fullfact_pyDOE.png)

- Box-Behnken DOE from pyDO
![Box-Behnken DOE from pyDOE](../_images/doe/bbdesign_pyDOE.png)

- LHS DOE from pyDOE
![LHS DOE from pyDOE](../_images/doe/lhs_pyDOE.png)

- Axial DOE from OpenTURNS
![Axial DOE from OpenTURNS](../_images/doe/axial_openturns.png)

- Composite DOE from OpenTURNS
![Composite DOE from OpenTURNS](../_images/doe/composite_openturns.png)

- Full Factorial DOE from OpenTURNS
![Full Factorial DOE from OpenTURNS](../_images/doe/factorial_openturns.png)

- Faure DOE from OpenTURNS
![Faure DOE from OpenTURNS](../_images/doe/faure_openturns.png)

- Halton DOE from OpenTURNS
![Halton DOE from OpenTURNS](../_images/doe/halton_openturns.png)

- Haselgrove DOE from OpenTURNS
![Haselgrove DOE from OpenTURNS](../_images/doe/haselgrove_openturns.png)

- Sobol DOE from OpenTURNS
![Sobol DOE from OpenTURNS](../_images/doe/sobol_openturns.png)

- Monte-Carlo DOE from OpenTURNS
![Monte-Carlo DOE from OpenTURNS](../_images/doe/mc_openturns.png)

- LHSC DOE from OpenTURNS
![LHSC DOE from OpenTURNS](../_images/doe/lhsc_openturns.png)

- LHS DOE from OpenTURNS
![LHS DOE from OpenTURNS](../_images/doe/lhs_openturns.png)

- Random DOE from OpenTURNS
![Random DOE from OpenTURNS](../_images/doe/random_openturns.png)
