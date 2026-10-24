---
status: new
description: "Migrating uncertainty from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Uncertainty

The changes of GEMSEO 7 about distributions, statistics and sensitivity analysis that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### `BaseSensitivityAnalysis.Method` was removed

`BaseSensitivityAnalysis.Method` was removed; the method enumeration of each sensitivity analysis is now defined outside its class, e.g. `CorrelationAnalysisMethod`, `HSICAnalysisMethod`, `MorrisAnalysisMethod` and `SobolAnalysisMethod`, which can be imported from `gemseo.enum`.

bump-gemseo leaves no comment, and `BaseSensitivityAnalysis.Method` raises `AttributeError: type object 'BaseSensitivityAnalysis' has no attribute 'Method'`. The base class has no replacement: use the enumeration of the analysis you work with, or remove the code.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: type object 'BaseSensitivityAnalysis' has no attribute 'Method'`
    - **See** issue [#1812](https://gitlab.com/gemseo/dev/gemseo/-/issues/1812), merge request [!2533](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2533).

=== "GEMSEO 6.3"

    ```python
    from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import BaseSensitivityAnalysis
    print(BaseSensitivityAnalysis.Method)
    ```

=== "GEMSEO 7"

    No GEMSEO 7 code is shown for this example: follow the explanation above.

### `DistributionFactory` creates distributions from settings

`DistributionFactory.create_marginal_distribution` and `DistributionFactory.create_joint_distribution` were removed; use `create_from_settings` instead, with the settings of a marginal distribution, e.g. `SPNormalDistribution_Settings(mu=1.0, sigma=2.0)`, or of a joint distribution, e.g. an `SPJointDistribution_Settings` whose `marginal_settings` field lists the settings of the marginals.

bump-gemseo leaves no comment, and the old methods raise `AttributeError: 'DistributionFactory' object has no attribute 'create_marginal_distribution'`. Replace the class name and keyword arguments by an instance of the settings class, and the list of marginal distributions by the list of their settings, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'DistributionFactory' object has no attribute 'create_marginal_distribution'`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2307](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2307).

=== "GEMSEO 6.3"

    ```python
    from gemseo.uncertainty.distributions.factory import DistributionFactory
    factory = DistributionFactory()
    first = factory.create_marginal_distribution("SPNormalDistribution", mu=1.0, sigma=2.0)
    second = factory.create_marginal_distribution("SPUniformDistribution", minimum=0.0, maximum=4.0)
    joint = factory.create_joint_distribution([first, second])
    print([first.mean, joint.mean])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.uncertainty.distribution.scipy.joint_settings import SPJointDistribution_Settings
    from gemseo.uncertainty.distribution import SPNormalDistribution_Settings
    from gemseo.uncertainty.distribution import SPUniformDistribution_Settings
    from gemseo.uncertainty.distribution.factory import DistributionFactory
    factory = DistributionFactory()
    first = factory.create_from_settings(SPNormalDistribution_Settings(mu=1.0, sigma=2.0))
    joint = factory.create_from_settings(SPJointDistribution_Settings(marginal_settings=[SPNormalDistribution_Settings(mu=1.0, sigma=2.0), SPUniformDistribution_Settings(minimum=0.0, maximum=4.0)]))
    print([first.mean, joint.mean])
    ```

### `IshigamiProblem` is no longer an optimization problem

`IshigamiProblem` derives from `EvaluationProblem` instead of `OptimizationProblem`. It is still defined over the `IshigamiSpace` and still takes a `uniform_distribution_name` argument, but the Ishigami function is now an observable of the problem instead of its objective.

bump-gemseo replaces `problem.objective` by `problem.observables[0]`, but leaves the other attributes specific to optimization problems without comment, and reading them raises an error, e.g. `AttributeError: 'IshigamiProblem' object has no attribute 'minimize_objective'`. GEMSEO 7 has no equivalent for these attributes on this problem: remove the code that uses them.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'IshigamiProblem' object has no attribute 'minimize_objective'`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.uncertainty.ishigami.ishigami_problem import IshigamiProblem
    problem = IshigamiProblem()
    print([problem.minimize_objective, len(problem.constraints)])
    ```

=== "GEMSEO 7"

    No GEMSEO 7 code is shown for this example: follow the explanation above.

### The Morris step must be smaller than 0.5

The `step` settings of `OATDOE` and `MorrisDOE`, also used by `MorrisAnalysis`, must be smaller than 0.5, so that the perturbed coordinate stays in the open interval $(0,1)$ on the probability scale; a greater step took this coordinate to 0, or out of the unit hypercube.

bump-gemseo leaves no comment, and a step of 0.5 or more raises `ValidationError: 1 validation error for MorrisDOE_Settings` (input should be less than 0.5). Choose a step smaller than 0.5, e.g. `step=0.49` as in the example below; the samples then differ from those of GEMSEO 6.3.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ValidationError: 1 validation error for MorrisDOE_Settings step Input should be less than 0.5 [type=less_than, input_value=0.5, input_type=float] For further information visit https://errors.pydantic.dev/2.14/v/less_than`
    - **See** issue [#1896](https://gitlab.com/gemseo/dev/gemseo/-/issues/1896), merge request [!2728](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2728).

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
    from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
    from gemseo.uncertainty.sensitivity.morris_analysis import MorrisAnalysis
    analysis = MorrisAnalysis()
    dataset = analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 0, n_replicates=2, step=0.5)
    print(len(dataset))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
    from gemseo.problem.uncertainty.ishigami import IshigamiSpace
    from gemseo.uncertainty.sensitivity import MorrisAnalysis
    analysis = MorrisAnalysis()
    dataset = analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 0, n_replicates=2, step=0.49)
    print(len(dataset))
    ```

### The OpenTURNS distributions take a settings object

`BaseDistribution` has a unique argument, `settings`. Each OpenTURNS distribution, e.g. `OTBetaDistribution` or `OTDistribution`, takes an instance of its settings class, e.g. `OTBetaDistribution_Settings`, whose fields are the former arguments; if `settings` is `None`, the default settings are used. `OTJointDistribution` takes an `OTJointDistribution_Settings`, whose `marginal_settings` field is the list of the settings of the marginal distributions, not the distributions themselves.

bump-gemseo leaves no comment, and the former arguments raise an error such as `TypeError: OTBetaDistribution.__init__() got an unexpected keyword argument 'alpha'`. Wrap the arguments of each distribution in its settings class, and build a joint distribution from the settings of its marginals, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: OTBetaDistribution.__init__() got an unexpected keyword argument 'alpha'`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2307](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2307).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.uncertainty.distributions.openturns.beta import OTBetaDistribution
    from gemseo.uncertainty.distributions.openturns.dirac import OTDiracDistribution
    from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
    from gemseo.uncertainty.distributions.openturns.exponential import OTExponentialDistribution
    from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
    from gemseo.uncertainty.distributions.openturns.log_normal import OTLogNormalDistribution
    from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution
    from gemseo.uncertainty.distributions.openturns.triangular import OTTriangularDistribution
    from gemseo.uncertainty.distributions.openturns.uniform import OTUniformDistribution
    from gemseo.uncertainty.distributions.openturns.weibull import OTWeibullDistribution
    marginals = [
        OTBetaDistribution(alpha=2.0, beta=3.0, minimum=0.0, maximum=2.0),
        OTDiracDistribution(variable_value=1.5),
        OTDistribution(interfaced_distribution="Gumbel", parameters=(1.0, 2.0)),
        OTExponentialDistribution(rate=2.0, loc=1.0),
        OTLogNormalDistribution(mu=1.0, sigma=0.5, location=0.5),
        OTNormalDistribution(mu=1.0, sigma=2.0),
        OTTriangularDistribution(minimum=0.0, mode=1.0, maximum=3.0),
        OTUniformDistribution(minimum=-1.0, maximum=3.0),
        OTWeibullDistribution(location=0.0, scale=2.0, shape=1.5),
    ]
    joint = OTJointDistribution([OTNormalDistribution(mu=1.0, sigma=2.0), OTUniformDistribution(minimum=-1.0, maximum=3.0)])
    print([[float(np.atleast_1d(m.mean)[0]) for m in marginals], joint.mean])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.uncertainty.distribution.openturns.beta import OTBetaDistribution
    from gemseo.uncertainty.distribution.openturns.beta_settings import OTBetaDistribution_Settings
    from gemseo.uncertainty.distribution.openturns.dirac import OTDiracDistribution
    from gemseo.uncertainty.distribution.openturns.dirac_settings import OTDiracDistribution_Settings
    from gemseo.uncertainty.distribution.openturns.distribution import OTDistribution
    from gemseo.uncertainty.distribution.openturns.distribution_settings import OTDistribution_Settings
    from gemseo.uncertainty.distribution.openturns.exponential import OTExponentialDistribution
    from gemseo.uncertainty.distribution.openturns.exponential_settings import OTExponentialDistribution_Settings
    from gemseo.uncertainty.distribution.openturns.joint import OTJointDistribution
    from gemseo.uncertainty.distribution.openturns.joint_settings import OTJointDistribution_Settings
    from gemseo.uncertainty.distribution.openturns.log_normal import OTLogNormalDistribution
    from gemseo.uncertainty.distribution.openturns.log_normal_settings import OTLogNormalDistribution_Settings
    from gemseo.uncertainty.distribution.openturns.normal import OTNormalDistribution
    from gemseo.uncertainty.distribution.openturns.normal_settings import OTNormalDistribution_Settings
    from gemseo.uncertainty.distribution.openturns.triangular import OTTriangularDistribution
    from gemseo.uncertainty.distribution.openturns.triangular_settings import OTTriangularDistribution_Settings
    from gemseo.uncertainty.distribution.openturns.uniform import OTUniformDistribution
    from gemseo.uncertainty.distribution.openturns.uniform_settings import OTUniformDistribution_Settings
    from gemseo.uncertainty.distribution.openturns.weibull import OTWeibullDistribution
    from gemseo.uncertainty.distribution.openturns.weibull_settings import OTWeibullDistribution_Settings
    marginals = [
        OTBetaDistribution(settings=OTBetaDistribution_Settings(alpha=2.0, beta=3.0, minimum=0.0, maximum=2.0)),
        OTDiracDistribution(settings=OTDiracDistribution_Settings(variable_value=1.5)),
        OTDistribution(settings=OTDistribution_Settings(interfaced_distribution="Gumbel", parameters=(1.0, 2.0))),
        OTExponentialDistribution(settings=OTExponentialDistribution_Settings(rate=2.0, loc=1.0)),
        OTLogNormalDistribution(settings=OTLogNormalDistribution_Settings(mu=1.0, sigma=0.5, location=0.5)),
        OTNormalDistribution(settings=OTNormalDistribution_Settings(mu=1.0, sigma=2.0)),
        OTTriangularDistribution(settings=OTTriangularDistribution_Settings(minimum=0.0, mode=1.0, maximum=3.0)),
        OTUniformDistribution(settings=OTUniformDistribution_Settings(minimum=-1.0, maximum=3.0)),
        OTWeibullDistribution(settings=OTWeibullDistribution_Settings(location=0.0, scale=2.0, shape=1.5)),
    ]
    joint = OTJointDistribution(OTJointDistribution_Settings(marginal_settings=[OTNormalDistribution_Settings(mu=1.0, sigma=2.0), OTUniformDistribution_Settings(minimum=-1.0, maximum=3.0)]))
    print([[float(np.atleast_1d(m.mean)[0]) for m in marginals], joint.mean])
    ```

### The SciPy distributions take a settings object

As for OpenTURNS, each SciPy distribution, e.g. `SPBetaDistribution` or `SPDistribution`, has a unique argument, `settings`, an instance of its settings class, e.g. `SPBetaDistribution_Settings`, whose fields are the former arguments. `SPJointDistribution` takes an `SPJointDistribution_Settings`, whose `marginal_settings` field is the list of the settings of the marginal distributions.

bump-gemseo leaves no comment, and the former arguments raise an error such as `TypeError: SPBetaDistribution.__init__() got an unexpected keyword argument 'alpha'`. Wrap the arguments of each distribution in its settings class, and build a joint distribution from the settings of its marginals, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: SPBetaDistribution.__init__() got an unexpected keyword argument 'alpha'`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2307](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2307).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.uncertainty.distributions.scipy.beta import SPBetaDistribution
    from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
    from gemseo.uncertainty.distributions.scipy.exponential import SPExponentialDistribution
    from gemseo.uncertainty.distributions.scipy.joint import SPJointDistribution
    from gemseo.uncertainty.distributions.scipy.log_normal import SPLogNormalDistribution
    from gemseo.uncertainty.distributions.scipy.normal import SPNormalDistribution
    from gemseo.uncertainty.distributions.scipy.triangular import SPTriangularDistribution
    from gemseo.uncertainty.distributions.scipy.uniform import SPUniformDistribution
    from gemseo.uncertainty.distributions.scipy.weibull import SPWeibullDistribution
    marginals = [
        SPBetaDistribution(alpha=2.0, beta=3.0, minimum=0.0, maximum=2.0),
        SPDistribution(interfaced_distribution="gumbel_r", parameters={"loc": 1.0, "scale": 2.0}),
        SPExponentialDistribution(rate=2.0, loc=1.0),
        SPLogNormalDistribution(mu=1.0, sigma=0.5, location=0.5),
        SPNormalDistribution(mu=1.0, sigma=2.0),
        SPTriangularDistribution(minimum=0.0, mode=1.0, maximum=3.0),
        SPUniformDistribution(minimum=-1.0, maximum=3.0),
        SPWeibullDistribution(location=0.0, scale=2.0, shape=1.5),
    ]
    joint = SPJointDistribution([SPNormalDistribution(mu=1.0, sigma=2.0), SPUniformDistribution(minimum=-1.0, maximum=3.0)])
    print([[float(np.atleast_1d(m.mean)[0]) for m in marginals], joint.mean])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.uncertainty.distribution.scipy.beta import SPBetaDistribution
    from gemseo.uncertainty.distribution.scipy.beta_settings import SPBetaDistribution_Settings
    from gemseo.uncertainty.distribution.scipy.distribution import SPDistribution
    from gemseo.uncertainty.distribution.scipy.distribution_settings import SPDistribution_Settings
    from gemseo.uncertainty.distribution.scipy.exponential import SPExponentialDistribution
    from gemseo.uncertainty.distribution.scipy.exponential_settings import SPExponentialDistribution_Settings
    from gemseo.uncertainty.distribution.scipy.joint import SPJointDistribution
    from gemseo.uncertainty.distribution.scipy.joint_settings import SPJointDistribution_Settings
    from gemseo.uncertainty.distribution.scipy.log_normal import SPLogNormalDistribution
    from gemseo.uncertainty.distribution.scipy.log_normal_settings import SPLogNormalDistribution_Settings
    from gemseo.uncertainty.distribution.scipy.normal import SPNormalDistribution
    from gemseo.uncertainty.distribution.scipy.normal_settings import SPNormalDistribution_Settings
    from gemseo.uncertainty.distribution.scipy.triangular import SPTriangularDistribution
    from gemseo.uncertainty.distribution.scipy.triangular_settings import SPTriangularDistribution_Settings
    from gemseo.uncertainty.distribution.scipy.uniform import SPUniformDistribution
    from gemseo.uncertainty.distribution.scipy.uniform_settings import SPUniformDistribution_Settings
    from gemseo.uncertainty.distribution.scipy.weibull import SPWeibullDistribution
    from gemseo.uncertainty.distribution.scipy.weibull_settings import SPWeibullDistribution_Settings
    marginals = [
        SPBetaDistribution(settings=SPBetaDistribution_Settings(alpha=2.0, beta=3.0, minimum=0.0, maximum=2.0)),
        SPDistribution(settings=SPDistribution_Settings(interfaced_distribution="gumbel_r", parameters={"loc": 1.0, "scale": 2.0})),
        SPExponentialDistribution(settings=SPExponentialDistribution_Settings(rate=2.0, loc=1.0)),
        SPLogNormalDistribution(settings=SPLogNormalDistribution_Settings(mu=1.0, sigma=0.5, location=0.5)),
        SPNormalDistribution(settings=SPNormalDistribution_Settings(mu=1.0, sigma=2.0)),
        SPTriangularDistribution(settings=SPTriangularDistribution_Settings(minimum=0.0, mode=1.0, maximum=3.0)),
        SPUniformDistribution(settings=SPUniformDistribution_Settings(minimum=-1.0, maximum=3.0)),
        SPWeibullDistribution(settings=SPWeibullDistribution_Settings(location=0.0, scale=2.0, shape=1.5)),
    ]
    joint = SPJointDistribution(SPJointDistribution_Settings(marginal_settings=[SPNormalDistribution_Settings(mu=1.0, sigma=2.0), SPUniformDistribution_Settings(minimum=-1.0, maximum=3.0)]))
    print([[float(np.atleast_1d(m.mean)[0]) for m in marginals], joint.mean])
    ```

### `ToleranceIntervalFactory` only accepts class names

`ToleranceIntervalFactory` only supports class names, for consistency with the other factories: use `"NormalToleranceInterval"` instead of `"Normal"`.

bump-gemseo leaves no comment, and a short name raises `ImportError: The class Normal is not available`, followed by the list of the valid names. Append `ToleranceInterval` to the short name, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ImportError: The class Normal is not available; the available ones are: ExponentialToleranceInterval, LogNormalToleranceInterval, NormalToleranceInterval, UniformToleranceInterval, WeibullMinToleranceInterval and ...`
    - **See** issue [#1412](https://gitlab.com/gemseo/dev/gemseo/-/issues/1412), merge request [!2403](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2403).

=== "GEMSEO 6.3"

    ```python
    from gemseo.uncertainty.statistics.tolerance_interval.factory import ToleranceIntervalFactory
    interval = ToleranceIntervalFactory().create("Normal", 30, mean=1.0, std=2.0)
    lower, upper = interval.compute(0.95, 0.9)
    print([lower, upper])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.uncertainty.statistic.tolerance_interval.factory import ToleranceIntervalFactory
    interval = ToleranceIntervalFactory().create("NormalToleranceInterval", 30, mean=1.0, std=2.0)
    lower, upper = interval.compute(0.95, 0.9)
    print([lower, upper])
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### `_create_distribution` takes the settings of the distribution

A distribution now takes a settings model, and `BaseDistribution._create_distribution` takes this settings model, `_create_distribution(settings)`, and reads the name and the parameters of the interfaced distribution from its fields, as `gemseo.uncertainty.distribution.scipy.distribution.SPDistribution` does.

bump-gemseo marks the method with a comment saying so. In the example, the distribution itself is still created with positional arguments, which raises `TypeError: BaseDistribution.__init__() takes from 1 to 2 positional arguments but 3 were given`. Pass an `OTDistribution_Settings` both to the constructor and to `_create_distribution`, as in the example below; a subclass that overrides `_create_distribution` must take the settings and read its fields.

!!! info "After bump-gemseo"

    - **Comment**: `_create_distribution now takes the settings model of the distribution, _create_distribution(settings), and reads the name and the parameters of the interfaced distribution from its fields, as gemseo.uncertainty.distribution.scipy.distribution.SPDistribution does`
    - **Error with GEMSEO 7**: `TypeError: BaseDistribution.__init__() takes from 1 to 2 positional arguments but 3 were given`

=== "GEMSEO 6.3"

    ```python
    from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
    distribution = OTDistribution("Normal", (0.0, 1.0))
    distribution._create_distribution("Uniform", (0.0, 2.0), "x", None, None, 0.5)
    print([distribution.mean, distribution.distribution.getMean()[0]])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.uncertainty.distribution import OTDistribution_Settings
    from gemseo.uncertainty.distribution.openturns.distribution import OTDistribution
    distribution = OTDistribution(OTDistribution_Settings(interfaced_distribution="Normal", parameters=(0.0, 1.0)))
    distribution._create_distribution(OTDistribution_Settings(interfaced_distribution="Uniform", parameters=(0.0, 2.0)))
    print([distribution.mean, distribution.distribution.getMean()[0]])
    ```

### The distribution fitters no longer take a variable name

The `variable` argument of `BaseDistributionFitter`, `OTDistributionFitter` and `SPDistributionFitter` was removed, and so was the `variable` attribute: the fitter no longer stores the variable name.

bump-gemseo marks the use of the attribute with the comment `BaseDistributionFitter.variable was removed; see the GEMSEO 7 changelog.`; left as is, reading it raises `AttributeError: 'OTDistributionFitter' object has no attribute 'variable'`. Create the fitter from the data only, e.g. `OTDistributionFitter(data)`, and keep the variable name in your own code if you need it, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `BaseDistributionFitter.variable was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'OTDistributionFitter' object has no attribute 'variable'`
    - **GEMSEO 7 cannot give the GEMSEO 6.3 value here**: the variable name is no longer stored by the fitter (issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)).
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2355](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2355).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.uncertainty.distributions.openturns.distribution_fitter import OTDistributionFitter
    data = np.random.default_rng(0).normal(2.0, 0.5, 200)
    fitter = OTDistributionFitter("x", data)
    distribution = fitter.fit("Normal")
    print([fitter.variable, distribution.mean])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.uncertainty.distribution.openturns.distribution_fitter import OTDistributionFitter
    data = np.random.default_rng(0).normal(2.0, 0.5, 200)
    fitter = OTDistributionFitter(data)
    distribution = fitter.fit("Normal")
    print(["<removed in GEMSEO 7>", distribution.mean])
    ```

### The distribution parameter type aliases were removed

The type aliases `ParametersType` and `StandardParametersType` of `gemseo.uncertainty.distributions.base_distribution` and `gemseo.uncertainty.distributions.scalar_distribution_mixin` were removed with no replacement; importing one raises an error saying so.

bump-gemseo marks each use with a comment giving the type to write instead: `tuple[str, int, float] | Mapping[str, str | int | float]` for `ParametersType`, and `Mapping[str, str | int | float]` for `StandardParametersType`, which is the type of the field `standard_parameters` of `BaseDistributionSettings`. Remove the import and write these types in your annotations, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `gemseo.uncertainty.distributions.base_distribution.ParametersType: use tuple[str, int, float] | Mapping[str, str | int | float] instead` / `gemseo.uncertainty.distributions.base_distribution.StandardParametersType: use Mapping[str, str | int | float] instead, the type of the field standard_parameters of BaseDistributionSettings` / `gemseo.uncertainty.distributions.scalar_distribution_mixin.ParametersType: use tuple[str, int, float] | Mapping[str, str | int | float] instead` / `gemseo.uncertainty.distributions.scalar_distribution_mixin.StandardParametersType: use Mapping[str, str | int | float] instead, the type of the field standard_parameters of BaseDistributionSettings`
    - **Error with GEMSEO 7**: `ImportError: The attribute 'ParametersType' of the module 'gemseo.uncertainty.distributions.base_distribution' was removed; use tuple[str, int, float] | Mapping[str, str | int | float] instead.`

=== "GEMSEO 6.3"

    ```python
    from gemseo.uncertainty.distributions import scalar_distribution_mixin
    from gemseo.uncertainty.distributions.base_distribution import ParametersType
    from gemseo.uncertainty.distributions.base_distribution import StandardParametersType
    print([str(ParametersType), str(StandardParametersType), str(scalar_distribution_mixin.ParametersType), str(scalar_distribution_mixin.StandardParametersType)])
    ```

=== "GEMSEO 7"

    ```python
    from collections.abc import Mapping
    parameters_type = tuple[str, int, float] | Mapping[str, str | int | float]
    standard_parameters_type = Mapping[str, str | int | float]
    print([str(parameters_type), str(standard_parameters_type), str(parameters_type), str(standard_parameters_type)])
    ```

### `HSICAnalysis` no longer computes permutation p-values by default

`HSICAnalysis.compute_indices` no longer estimates the p-values through permutations by default, because this estimation costs `n_permutations` HSIC estimations and used to dominate the computation time. Set its new `use_permutations` argument to `True` to do so; otherwise `indices.p_value_permutation` is empty, and `HSICAnalysis.main_indices` raises a `ValueError` when `HSICAnalysis.main_method` is `p-value-permutation`.

bump-gemseo does not handle this change. In the example, it marks `compute_samples` with the comment `cannot transform: algo_settings is passed without algo, whose default names no Settings class`; left as is, it raises `AttributeError: 'dict' object has no attribute 'n_samples'`. Pass a DOE settings object, e.g. `OT_MONTE_CARLO_Settings(seed=3)`; the code then runs, but without p-values by permutation. Add `use_permutations=True` wherever you use these p-values, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: algo_settings is passed without algo, whose default names no Settings class`
    - **Error with GEMSEO 7**: `AttributeError: 'dict' object has no attribute 'n_samples'`
    - **See** issue [#1908](https://gitlab.com/gemseo/dev/gemseo/-/issues/1908), merge request [!2738](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2738).

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
    from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
    from gemseo.uncertainty.sensitivity.hsic_analysis import HSICAnalysis
    analysis = HSICAnalysis()
    analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 40, algo_settings={"seed": 3})
    indices = analysis.compute_indices(n_permutations=20, seed=5)
    permutation = indices.p_value_permutation
    print([sorted(permutation), [len(component) for component in permutation["y"]]])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import OT_MONTE_CARLO_Settings
    from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
    from gemseo.problem.uncertainty.ishigami import IshigamiSpace
    from gemseo.uncertainty.sensitivity import HSICAnalysis
    analysis = HSICAnalysis()
    analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 40, algo_settings=OT_MONTE_CARLO_Settings(seed=3))
    indices = analysis.compute_indices(n_permutations=20, seed=5, use_permutations=True)
    permutation = indices.p_value_permutation
    print([sorted(permutation), [len(component) for component in permutation["y"]]])
    ```

### The plots of the sensitivity analyses take settings objects

The `**settings` of `BaseSensitivityAnalysis.plot_bar` were replaced by `bar_plot_settings: BarPlot_Settings | None = None`. The `**settings` of `BaseSensitivityAnalysis.plot_radar` were replaced by `radar_chart_settings: RadarChart_Settings | None = None`, and its `min_radius` and `max_radius` arguments were replaced by the `rmin` and `rmax` fields of `RadarChart_Settings` (from `gemseo.post.dataset.radar_chart_settings`). The `**settings` of `BaseSensitivityAnalysis.plot_comparison` were removed.

bump-gemseo does not rewrite these calls. In the example, it marks `compute_samples` with the comment `cannot transform: algo_settings is passed without algo, whose default names no Settings class`, because `algo_settings` must now be a `BaseDOESettings`, not a `dict`; left as is, it raises `AttributeError: 'dict' object has no attribute 'n_samples'`. Pass a DOE settings object, e.g. `OT_MONTE_CARLO_Settings(seed=1)`, then move the plot options into `BarPlot_Settings` and `RadarChart_Settings`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: algo_settings is passed without algo, whose default names no Settings class`
    - **Error with GEMSEO 7**: `AttributeError: 'dict' object has no attribute 'n_samples'`
    - **See** merge request [!2403](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2403).

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
    from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
    from gemseo.uncertainty.sensitivity.correlation_analysis import CorrelationAnalysis
    analysis = CorrelationAnalysis()
    analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 30, algo_settings={"seed": 1})
    analysis.compute_indices()
    bars = analysis.plot_bar(save=False, n_digits=3)
    radar = analysis.plot_radar(save=False, min_radius=-1.0, max_radius=1.0)
    other = CorrelationAnalysis()
    other.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 30, algo_settings={"seed": 2})
    other.compute_indices()
    comparison = analysis.plot_comparison(other, "y", save=False)
    print([type(bars).__name__, type(radar).__name__, type(comparison).__name__])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import OT_MONTE_CARLO_Settings
    from gemseo.post.dataset.bar_plot_settings import BarPlot_Settings
    from gemseo.post.dataset.radar_chart_settings import RadarChart_Settings
    from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
    from gemseo.problem.uncertainty.ishigami import IshigamiSpace
    from gemseo.uncertainty.sensitivity import CorrelationAnalysis
    analysis = CorrelationAnalysis()
    analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 30, algo_settings=OT_MONTE_CARLO_Settings(seed=1))
    analysis.compute_indices()
    bars = analysis.plot_bar(save=False, bar_plot_settings=BarPlot_Settings(n_digits=3))
    radar = analysis.plot_radar(save=False, radar_chart_settings=RadarChart_Settings(rmin=-1.0, rmax=1.0))
    other = CorrelationAnalysis()
    other.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 30, algo_settings=OT_MONTE_CARLO_Settings(seed=2))
    other.compute_indices()
    comparison = analysis.plot_comparison(other, "y", save=False)
    print([type(bars).__name__, type(radar).__name__, type(comparison).__name__])
    ```

### The Sobol' samples store the space under `random_space`

`SobolAnalysis.compute_samples` stores the random space it was given in `dataset.misc["random_space"]` instead of `dataset.misc["parameter_space"]`.

bump-gemseo cannot infer the type of `dataset.misc["parameter_space"]` and marks the line with a comment starting with `cannot transform: the type of dataset.misc["parameter_space"] could not be inferred`; left as is, the line raises `KeyError: 'parameter_space'`. Read `dataset.misc["random_space"]` instead and get the variable names from its `variables` attribute rather than `variable_names`: in the comment, `variable_names became list({receiver}.variables)` means `list(dataset.misc["random_space"].variables)` here, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: the type of dataset.misc["parameter_space"] could not be inferred; if it is an instance of DesignSpace, variable_names became list({receiver}.variables)`
    - **Error with GEMSEO 7**: `KeyError: 'parameter_space'`
    - **See** merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
    from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
    from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis
    analysis = SobolAnalysis()
    dataset = analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 40, compute_second_order=False)
    print(sorted(dataset.misc["parameter_space"].variable_names))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
    from gemseo.problem.uncertainty.ishigami import IshigamiSpace
    from gemseo.uncertainty.sensitivity import SobolAnalysis
    analysis = SobolAnalysis()
    dataset = analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 40, compute_second_order=False)
    print(sorted(dataset.misc["random_space"].variables))
    ```

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. These names have no direct replacement: unless an entry above or the changelog gives one, remove the code that uses them.

??? note "Private names (32), used by some plugins"

    - `gemseo.uncertainty.distributions.base_settings.beta_settings._ALPHA`
    - `gemseo.uncertainty.distributions.base_settings.beta_settings._BETA`
    - `gemseo.uncertainty.distributions.base_settings.beta_settings._MAXIMUM`
    - `gemseo.uncertainty.distributions.base_settings.beta_settings._MINIMUM`
    - `gemseo.uncertainty.distributions.base_settings.exponential_settings._LOC`
    - `gemseo.uncertainty.distributions.base_settings.exponential_settings._RATE`
    - `gemseo.uncertainty.distributions.base_settings.log_normal_settings._LOCATION`
    - `gemseo.uncertainty.distributions.base_settings.log_normal_settings._MU`
    - `gemseo.uncertainty.distributions.base_settings.log_normal_settings._SET_LOG`
    - `gemseo.uncertainty.distributions.base_settings.log_normal_settings._SIGMA`
    - `gemseo.uncertainty.distributions.base_settings.normal_settings._MU`
    - `gemseo.uncertainty.distributions.base_settings.normal_settings._SIGMA`
    - `gemseo.uncertainty.distributions.base_settings.triangular_settings._MAXIMUM`
    - `gemseo.uncertainty.distributions.base_settings.triangular_settings._MINIMUM`
    - `gemseo.uncertainty.distributions.base_settings.triangular_settings._MODE`
    - `gemseo.uncertainty.distributions.base_settings.uniform_settings._MAXIMUM`
    - `gemseo.uncertainty.distributions.base_settings.uniform_settings._MINIMUM`
    - `gemseo.uncertainty.distributions.base_settings.weibull_settings._LOCATION`
    - `gemseo.uncertainty.distributions.base_settings.weibull_settings._SCALE`
    - `gemseo.uncertainty.distributions.base_settings.weibull_settings._SHAPE`
    - `gemseo.uncertainty.distributions.base_settings.weibull_settings._USE_WEIBULL_MIN`
    - `gemseo.uncertainty.distributions.openturns.distribution_settings._INTERFACED_DISTRIBUTION`
    - `gemseo.uncertainty.distributions.openturns.distribution_settings._LOWER_BOUND`
    - `gemseo.uncertainty.distributions.openturns.distribution_settings._OTDistribution_Settings_Mixin`
    - `gemseo.uncertainty.distributions.openturns.distribution_settings._PARAMETERS`
    - `gemseo.uncertainty.distributions.openturns.distribution_settings._STANDARD_PARAMETERS`
    - `gemseo.uncertainty.distributions.openturns.distribution_settings._THRESHOLD`
    - `gemseo.uncertainty.distributions.openturns.distribution_settings._TRANSFORMATION`
    - `gemseo.uncertainty.distributions.openturns.distribution_settings._UPPER_BOUND`
    - `gemseo.uncertainty.distributions.scipy.distribution_settings._INTERFACED_DISTRIBUTION`
    - `gemseo.uncertainty.distributions.scipy.distribution_settings._PARAMETERS`
    - `gemseo.uncertainty.distributions.scipy.distribution_settings._STANDARD_PARAMETERS`
