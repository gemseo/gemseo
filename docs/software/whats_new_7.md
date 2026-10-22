---
status: draft
description: ""
tags: []
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

# What's new in GEMSEO 7

GEMSEO 7 is the biggest release in the history of the library.
We went through every corner of the API with one question in mind:
*what would make your next study faster to write, safer to run and easier to share?*
The answer is a GEMSEO that catches your mistakes before the computation starts,
speaks the language of uncertainty as fluently as the language of optimization,
lets you design with real-world choices such as materials and catalog parts,
and runs some workflows up to twenty times faster.

```shell
pip install --upgrade gemseo[all]
```

<div class="grid cards" markdown>

-   :material-shield-check:{ .lg .middle } **Mistakes caught at once**

    ---

    Every algorithm, MDA, formulation and post-processor is configured
    with a typed settings object, checked as soon as you write it.

-   :material-dice-multiple:{ .lg .middle } **Uncertainty, first-class**

    ---

    A dedicated random space, dependent variables through copulas,
    reliability analysis and new sensitivity analyses.

-   :material-shape:{ .lg .middle } **Design with real choices**

    ---

    Discrete values, categorical labels and catalog parts
    join the real and integer design variables.

-   :material-rocket-launch:{ .lg .middle } **Faster**

    ---

    A DOE filling an HDF5 cache now takes 4 seconds instead of 78,
    and databases no longer slow down as they grow.

</div>

## Your settings are checked before your computation starts

Long studies used to fail in the worst possible way:
a misspelled option such as `max_itre` was silently ignored,
and you found out hours later that your optimizer had stopped after its default number of iterations.

In GEMSEO 7, the algorithms, the MDAs, the MDO formulations, the post-processors,
the dataset plots, the probability distributions and the machine learning models
are all configured with a [Pydantic](https://docs.pydantic.dev) settings object.
A typo or a value of the wrong type is rejected the moment you write it,
and again whenever you modify a setting afterwards.
Because each settings class lists its fields with their types, defaults and descriptions,
your IDE completes them for you and shows their documentation as you type:
no more back-and-forth with the documentation to remember an option name.

```python
from pydantic import ValidationError

from gemseo.optimization import SLSQP_Settings

try:
    SLSQP_Settings(max_itre=100)
except ValidationError as error:
    print(error)

settings = SLSQP_Settings(max_iter=100)
try:
    settings.max_iter = "many"
except ValidationError as error:
    print("Rejected on assignment.")
```

Settings also nest naturally.
Instead of naming an MDA with a string and passing its options in a dictionary,
you give the formulation the settings of its MDA,
which carry the settings of their linear solver, and so on:
the whole configuration of a process is visible, and checked, in one place.

## Scenarios that read like your problem

Building a scenario is now a sentence:
*take these disciplines, this design space and this formulation;
minimize this, subject to that;
run this algorithm.*
The objective is no longer frozen in the constructor:
`add_objective` declares it like the constraints and observables,
and can maximize it or combine a group of objectives.
`execute` hands you back the result of the algorithm,
so the optimum is one variable away instead of a lookup in the optimization problem.

=== "GEMSEO 6"

    ```python
    from gemseo import create_discipline
    from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
    from gemseo.scenarios.mdo_scenario import MDOScenario

    scenario = MDOScenario(
        create_discipline(["Sellar1", "Sellar2", "SellarSystem"]),
        "obj",
        SellarDesignSpace(),
        formulation_name="MDF",
        main_mda_name="MDAGaussSeidel",
        main_mda_settings={"tolerance": 1e-8},
    )
    scenario.add_constraint("c_1", constraint_type="ineq")
    scenario.add_constraint("c_2", constraint_type="ineq")
    scenario.execute(algo_name="SLSQP", max_iter=100)
    scenario.post_process(post_name="OptHistoryView", save=False, show=False)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.formulation import MDF_Settings
    from gemseo.mda import MDAGaussSeidel_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.post import OptHistoryView_Settings
    from gemseo.problem.mdo.sellar import Sellar1, Sellar2, SellarDesignSpace, SellarSystem
    from gemseo.scenario import MDOScenario

    scenario = MDOScenario(
        [Sellar1(), Sellar2(), SellarSystem()],
        SellarDesignSpace(),
        formulation_settings=MDF_Settings(
            main_mda_settings=MDAGaussSeidel_Settings(tolerance=1e-8)
        ),
    )
    scenario.add_objective("obj")
    scenario.add_constraint("c_1", constraint_type="ineq")
    scenario.add_constraint("c_2", constraint_type="ineq")
    result = scenario.execute(SLSQP_Settings(max_iter=100))
    scenario.post_process(OptHistoryView_Settings(save=False, show=False))
    ```

One scenario class now does the job of two.
Optimization or design of experiments, an `MDOScenario` simply runs the algorithm you give it,
so switching from a DOE to an optimizer is a one-line change:

=== "GEMSEO 6"

    ```python
    from gemseo import create_discipline
    from gemseo.algos.design_space import DesignSpace
    from gemseo.scenarios.doe_scenario import DOEScenario

    discipline = create_discipline("AnalyticDiscipline", expressions={"y": "x1+x2"})
    design_space = DesignSpace()
    design_space.add_variable("x1", lower_bound=0.0, upper_bound=1.0)
    design_space.add_variable("x2", lower_bound=0.0, upper_bound=1.0)

    scenario = DOEScenario(
        [discipline], "y", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=25)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.doe import PYDOE_FULLFACT_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.space import DesignSpace

    discipline = AnalyticDiscipline({"y": "x1+x2"})
    design_space = DesignSpace()
    design_space.add_real_variable("x1", lower_bound=0.0, upper_bound=1.0)
    design_space.add_real_variable("x2", lower_bound=0.0, upper_bound=1.0)

    scenario = MDOScenario([discipline], design_space)
    scenario.add_objective("y")
    scenario.execute(PYDOE_FULLFACT_Settings(n_samples=25))
    ```

And when you only want to explore,
the new `EvaluationScenario` samples your disciplines without asking for an objective;
`observe_all_outputs` records everything they compute:

```python
from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import PYDOE_FULLFACT_Settings
from gemseo.scenario import EvaluationScenario
from gemseo.space import DesignSpace

discipline = AnalyticDiscipline({"y": "x1+x2", "z": "x1*x2"})
design_space = DesignSpace()
design_space.add_real_variable("x1", lower_bound=0.0, upper_bound=1.0)
design_space.add_real_variable("x2", lower_bound=0.0, upper_bound=1.0)

scenario = EvaluationScenario([discipline], design_space)
scenario.observe_all_outputs()
scenario.execute(PYDOE_FULLFACT_Settings(n_samples=25))
dataset = scenario.to_dataset()
```

Scenarios also travel further.
`wrap_scenario_in_job_scheduler` sends a whole scenario to an HPC cluster in one call,
and bi-level studies gain flexibility:
plug in your own MDAs, use plain disciplines as sub-scenarios,
and retrieve the full history of a sub-scenario across all its executions as a single dataset.

## Know where every file goes

Simulations that wrap executables leave files behind,
and finding the ones of iteration 42 among thousands of folders used to be an archaeology job.
[The directory manager][the-directory-manager] organizes them for you
into a tree that mirrors your workflow:
one directory per scenario, per iteration or sample, per MDA iteration and per discipline execution,
each with a trace of what ran there.
Turn it on with a single line, `configuration.directory_manager.enable = True`,
and debugging a failed evaluation becomes a matter of opening the right folder.

## Uncertainty is now a first-class citizen

### A space made for random variables

GEMSEO 6 described random variables with a `ParameterSpace`,
a design space with probability distributions bolted on,
where bounds, current values and normalization made little sense.
GEMSEO 7 gives them a [RandomSpace][concept-random-space] of their own,
defined by probability distributions and nothing else.
Design spaces and random spaces share a common parent,
so the DOE algorithms, the scenarios, the formulations and the sensitivity analyses
accept either one without any conversion.

=== "GEMSEO 6"

    ```python
    from gemseo import create_discipline, sample_disciplines
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.settings.probability_distributions import SPNormalDistribution_Settings

    discipline = create_discipline("AnalyticDiscipline", expressions={"z": "x+y"})
    space = ParameterSpace()
    space.add_random_variable("x", SPNormalDistribution_Settings(mu=0.0, sigma=1.0))
    space.add_random_variable("y", SPNormalDistribution_Settings(mu=0.0, sigma=1.0))
    dataset = sample_disciplines(
        [discipline], space, "z", algo_name="PYDOE_LHS", n_samples=5
    )
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.discipline import AnalyticDiscipline
    from gemseo.doe import PYDOE_LHS_Settings
    from gemseo.scenario import EvaluationScenario
    from gemseo.space import RandomSpace
    from gemseo.uncertainty.distribution import SPNormalDistribution_Settings

    discipline = AnalyticDiscipline({"z": "x+y"})
    space = RandomSpace()
    space.add_variable("x", SPNormalDistribution_Settings(mu=0.0, sigma=1.0))
    space.add_variable("y", SPNormalDistribution_Settings(mu=0.0, sigma=1.0))
    scenario = EvaluationScenario([discipline], space)
    scenario.add_observable("z")
    scenario.execute(PYDOE_LHS_Settings(n_samples=5))
    dataset = scenario.to_dataset()
    ```

### Dependent variables, at last

Real-world uncertainties are rarely independent:
the yield strength and the Young's modulus of a material,
the temperature and the pressure of a flow, move together.
Ignoring that dependence can make a design look safer than it is.
With GEMSEO 7, you describe it with a copula,
and every sample drawn from the space,
for a DOE as for a sensitivity analysis, respects it.
The pair plot can even display the ranks of your data,
so you can see the dependence you are modeling.

```python
from openturns import CorrelationMatrix, NormalCopula

from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import PYDOE_LHS_Settings
from gemseo.scenario import EvaluationScenario
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings

space = RandomSpace()
space.add_variable("x", OTNormalDistribution_Settings(mu=0.0, sigma=1.0))
space.add_variable("y", OTNormalDistribution_Settings(mu=0.0, sigma=1.0))
correlation = CorrelationMatrix(2)
correlation[0, 1] = 0.8
space.add_copula(("x", "y"), NormalCopula(correlation))
print(space)

scenario = EvaluationScenario([AnalyticDiscipline({"z": "x+y"})], space)
scenario.add_observable("z")
scenario.execute(PYDOE_LHS_Settings(n_samples=100))
dataset = scenario.to_dataset()
print(dataset.get_view(variable_names=["x", "y"]).corr())
```

### How likely is it to fail?

Knowing the mean and the variance of an output is not enough
when what matters is the probability of exceeding a threshold.
The new reliability analysis package answers exactly that question.
Define a failure event from thresholds on the outputs of your disciplines,
and estimate its probability with the methods of OpenTURNS:
FORM and SORM when the evaluations are expensive,
Monte Carlo, quasi Monte Carlo, subset, importance or directional sampling when the event is rare.

Sensitivity analysis follows the same path.
`ROSobolAnalysis` tells which inputs drive a [rare event][concept-rosobol-analysis],
`FORMAnalysis` turns the importance factors of FORM into sensitivity indices,
and `SobolAnalysis` can now estimate its first-order indices
[from independent and identically distributed samples][concept-sobol-estimators],
while `plot_second_order` reveals the interactions between inputs as a heat map.

## Design with real-world choices

Engineers do not only choose numbers in an interval.
They pick a diameter from a list of standard sizes,
a material among steel, aluminium and titanium,
a motor from a supplier's catalog.
GEMSEO 7 lets you say exactly that.
[Discrete variables][concept-discrete-variables] take their values in a finite list of numbers,
categorical variables take a label, and the disciplines receive the label itself,
and [catalog variables][concept-catalog-variables] pick a row in a table of alternatives.
Each kind has its own method, whose name tells what you declare and whose arguments are checked on the spot:

```python
from gemseo.space import DesignSpace

space = DesignSpace()
space.add_real_variable("thickness", lower_bound=0.1, upper_bound=1.0)
space.add_integer_variable("n_blades", lower_bound=2, upper_bound=8)
space.add_discrete_variable("diameter", (1.0, 4.0, 6.0, 9.0))
space.add_categorical_variable("material", ("steel", "aluminium", "titanium"))
```

GEMSEO 7 is also honest about what an algorithm can do with them.
An optimizer that only handles real numbers no longer rounds your integer variables behind your back:
it tells you so, and you decide whether to
[relax them into real variables][concept-integer-relaxation].
The result even reports the optimum projected back onto the domain you declared,
with its objective value and its feasibility.

## Faster where it hurts

Some workflows got dramatically faster:

- **Caches.** Filling a cache that stores every evaluation used to cost more with every new entry.
  It now finds an entry directly from its hash:
  on a DOE of 20 samples of the SSBJ problem,
  4 seconds instead of 78 with an `HDF5Cache`, and 13 instead of 25 with a `MemoryFullCache`.
- **Databases.** Exporting the history of an optimization to a dataset,
  or reading its last iterations, no longer slows down as the history grows.
- **Sensitivity analysis.** `HSICAnalysis` skips its permutation tests by default,
  which used to dominate its computation time.
- **Start-up.** Importing a package no longer imports everything it contains:
  the modules are loaded when you first use them.
- **Fast mode.** One call, `configuration.enable_fast_mode()`,
  switches off the checks and statistics that cost time at every evaluation;
  see [fast mode][global-configuration-fast].

## Results you can trust

A library for engineering decisions must be right before being fast.
GEMSEO 7 fixes several issues that could silently change your results.
Disciplines no longer receive input values truncated to the size of their default values.
Jacobians are now correct in edge cases of discipline chains
and when the input sizes differ from the default ones.
A constraint evaluated to NaN is no longer counted as satisfied,
and an infinite constraint value can no longer make the optimizer stop at an infeasible point.
The Morris sensitivity indices use all the replicates and the right estimators.

Checking your own gradients is easier too:
`gemseo.check_jacobian()` compares the derivatives of a discipline
with a numerical approximation in one call,
and discipline chains can now propagate derivatives forward as well as backward,
choosing the cheaper direction automatically.

## Everything where you expect it

GEMSEO 7 reorganizes its packages by domain,
`gemseo.optimization`, `gemseo.doe`, `gemseo.mda`, `gemseo.formulation`, `gemseo.scenario`,
`gemseo.space` and so on,
and every package offers its public classes, settings and factories directly.
You no longer need to know which module defines a class to import it:

=== "GEMSEO 6"

    ```python
    from gemseo.algos.design_space import DesignSpace
    from gemseo.scenarios.mdo_scenario import MDOScenario
    from gemseo.settings.formulations import MDF_Settings
    from gemseo.settings.opt import SLSQP_Settings
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.formulation import MDF_Settings
    from gemseo.optimization import SLSQP_Settings
    from gemseo.scenario import MDOScenario
    from gemseo.space import DesignSpace
    ```

Names follow the same rules everywhere, so you can often guess them:
`MDOFunction` becomes `ArrayFunction`, the chains of disciplines become `DisciplineChain`,
enumeration members are in upper case and the mappings are named `key_to_value`.
The machine learning models follow the same settings-based pattern as the rest of the library:

=== "GEMSEO 6"

    ```python
    from gemseo.mlearning.regression.algos.linreg import LinearRegressor

    regressor = LinearRegressor(dataset, fit_intercept=False)
    ```

=== "GEMSEO 7"

    ```python
    from numpy import linspace

    from gemseo.dataset import IODataset
    from gemseo.machine_learning import LinearRegressor_Settings
    from gemseo.machine_learning.regression.model import LinearRegressor

    x = linspace(0.0, 1.0, 10)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", 2.0 * x + 1.0)
    regressor = LinearRegressor(
        dataset, settings=LinearRegressor_Settings(fit_intercept=False)
    )
    regressor.learn()
    ```

## Upgrading is easier than it looks

A release of this size changes many names, and we made sure you do not pay for it.
Almost every renamed module, class, function and attribute keeps working under its old name
for a full deprecation cycle, with a `DeprecationWarning` telling you its replacement:
your GEMSEO 6 scripts keep running while you migrate.
[bump-gemseo](https://gitlab.com/gemseo/dev/bump-gemseo) rewrites most of your code automatically,
and the [Upgrading GEMSEO][upgrading-gemseo] page covers the rest, case by case.

GEMSEO 7 requires Python 3.11 or later
and raises the minimum versions of several dependencies,
including OpenTURNS 1.27, Pydantic 2.12, SciPy 1.15 and pandas 2.2.
The [changelog](../changelog.md) lists every change of the release.

Ready? Upgrade, run your scripts with the warnings on, and enjoy GEMSEO 7:

```shell
pip install --upgrade gemseo[all]
```
