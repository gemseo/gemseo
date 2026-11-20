---
reading_time: true
complexity: advanced
description: "Uncertainty characterization represents each uncertain model input as a probability distribution."
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

# Uncertainty characterization { #concept-uncertainty-characterization }

Uncertainty characterization is the first step of a UQ study:
it consists in assigning a [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)
(that we call it *distribution* for brevity)
to each uncertain input variable.
The uncertain variable is thus represented by a [random variable](https://en.wikipedia.org/wiki/Random_variable).
A distribution is a mathematical object
that describes the relative likelihood of each possible value that the variable can take.
The integral (or sum in the case of a discrete variable) of all values is 1.
For example,
when rolling an unloaded die,
the possible outcomes are 1, 2, ..., 6 and each outcome has a probability of occurrence equal to 1/6.

GEMSEO allows you to characterize uncertain scalar variables as random variables,
using univariate distributions,
and uncertain vector variables as random vectors,
using joint distributions.

## Univariate distributions

GEMSEO has a [BaseUnivariateDistribution][gemseo.uncertainty.distribution.core.base_univariate.BaseUnivariateDistribution] class
to wrap univariate distributions from external libraries.
The interface offers many services:
querying the probability density function (PDF) and cumulative distribution function (CDF),
plotting the PDF and CDF,
sampling from the distribution and
computing statistical moments (mean, standard deviation).

The available backends are
[OpenTURNS](https://openturns.github.io/www/) and [SciPy](https://docs.scipy.org/doc/scipy)
with the associated classes
[OTDistribution][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution]
and [SPDistribution][gemseo.uncertainty.distribution.scipy.distribution.SPDistribution].
Any distribution available in these external libraries can be created from these classes.
Subclasses also exist for classical distributions,
such as
[OTUniformDistribution][gemseo.uncertainty.distribution.openturns.uniform.OTUniformDistribution],
[OTNormalDistribution][gemseo.uncertainty.distribution.openturns.normal.OTNormalDistribution]
and
[OTTriangularDistribution][gemseo.uncertainty.distribution.openturns.triangular.OTTriangularDistribution]
for OpenTURNS,
and
[SPUniformDistribution][gemseo.uncertainty.distribution.scipy.uniform.SPUniformDistribution],
[SPNormalDistribution][gemseo.uncertainty.distribution.scipy.normal.SPNormalDistribution]
and
[SPTriangularDistribution][gemseo.uncertainty.distribution.scipy.triangular.SPTriangularDistribution]
for SciPy.

These distributions are defined from settings,
e.g. [OTUniformDistribution_Settings][gemseo.uncertainty.distribution.openturns.uniform_settings.OTUniformDistribution_Settings]
in the case of the [OTUniformDistribution][gemseo.uncertainty.distribution.openturns.uniform.OTUniformDistribution].

## Choosing a distribution { #concept-choosing-a-distribution }

The choice of distribution depends on whether data are available.

### Without data { #concept-choosing-a-distribution-without-data }

Without data, expert knowledge can be translated into a distribution
by the [maximum entropy principle](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy) [@jaynes1957]:
among all distributions consistent with the available information,
choose the one with the highest Shannon entropy,
i.e., the least informative one.
Common distributions are catalogued in [@kapur1989].

The mappings are:

| Prior knowledge                                     | Distribution                                                   | Reference                  |
|-----------------------------------------------------|----------------------------------------------------------------|----------------------------|
| Bounds $[a, b]$                                     | Uniform $\mathcal{U}(a, b)$                                    | [@lisman1972]              |
| Mean $\mu$ and standard deviation $\sigma$          | Normal $\mathcal{N}(\mu, \sigma^2)$                            | [@lisman1972]              |
| Lower bound $a$ and mean $\mu$                      | Exponential with rate $(\mu-a)^{-1}$, starting at $a$          | [@cover2006], section 12.2 |
| Upper bound $b$ and mean $\mu$                      | Reflected exponential with rate $(b-\mu)^{-1}$, ending at $b$  | [@cover2006], section 12.2 |
| Bounds $[a, b]$ and mean $\mu$                      | Uniform tilted by $e^{\lambda x}$ to have the mean $\mu$           | [@conrad], theorem 5.1     |
| Bounds (one or two), mean $\mu$, standard deviation $\sigma$ | Normal truncated to these bounds, for $\sigma$ not larger than the standard deviation of the row above matching the bounds | [@dowson1973]              |

Rather than selecting the distribution yourself,
state what you know to a maximum-entropy distribution,
which recognizes the case and applies the mapping:

```python
from gemseo.uncertainty.distribution import OTMaximumEntropyDistribution_Settings

# Only bounds are known.
settings = OTMaximumEntropyDistribution_Settings(minimum=a, maximum=b)
# A lower bound and a mean are known.
settings = OTMaximumEntropyDistribution_Settings(minimum=a, mu=mu)
```

These settings can be passed to
[OTMaximumEntropyDistribution][gemseo.uncertainty.distribution.openturns.maximum_entropy.OTMaximumEntropyDistribution]
or directly to
[RandomSpace.add_variable][gemseo.space.random.RandomSpace.add_variable].
Its SciPy counterpart
[SPMaximumEntropyDistribution][gemseo.uncertainty.distribution.scipy.maximum_entropy.SPMaximumEntropyDistribution]
takes
[SPMaximumEntropyDistribution_Settings][gemseo.uncertainty.distribution.scipy.maximum_entropy_settings.SPMaximumEntropyDistribution_Settings],
which have the same fields.
Any other combination of prior knowledge raises an error explaining what to add,
as no distribution maximizes the entropy for it.

!!! warning "The triangular distribution does not maximize the entropy"

    When an expert knows a minimum, a maximum and a most likely value $c$,
    engineering practice often uses the triangular distribution
    $\mathcal{T}(a, b, c)$.
    This distribution does **not** maximize the entropy:
    no maximum-entropy distribution is characterized by a most likely value,
    and assuming one adds information that the entropy principle cannot justify.
    Use
    [OTTriangularDistribution][gemseo.uncertainty.distribution.openturns.triangular.OTTriangularDistribution]
    to make this assumption deliberately,
    knowing that it is not the least informative choice.

Two of these cases are supported only by OpenTURNS,
as SciPy implements neither the reflected exponential distribution
nor the tilted uniform distribution whose density function increases.

### With data { #concept-choosing-a-distribution-with-data }

With data,
the parameters of a distribution can be estimated by a statistical technique,
such as the [maximum likelihood method](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
or the [method of moments](https://en.wikipedia.org/wiki/Method_of_moments_(statistics)).

[OTDistributionFitter][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter]
automatically selects the best-fitting distribution from a set of candidates
using this kind of technique.
It tests each candidate using one of these goodness-of-fit criteria:
the [Bayesian information criterion (BIC)](https://en.wikipedia.org/wiki/Bayesian_information_criterion),
the [Kolmogorov–Smirnov statistical hypothesis test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
or the [Chi-squared statistical hypothesis test](https://en.wikipedia.org/wiki/Chi-squared_test),
and returns the distribution that best matches the data.

In the case of SciPy,
[SPDistributionFitter][gemseo.uncertainty.distribution.scipy.distribution_fitter.SPDistributionFitter]
uses one of these statistical hypothesis tests as goodness-of-fit criterion:
[Anderson-Darling](https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test),
[Cramer-von-Mises](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion),
or [Kolmogorov–Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test).

These classes allow you to overlay the curves of different distributions on the data histogram,
providing a better understanding of the quality of the fit and the selections made.

## Joint distributions { #concept-joint-distributions }

A random vector requires a *joint* distribution
that describes its components collectively.
These components are characterized by univariate distributions,
called marginal distributions.
When the components are [independent](https://en.wikipedia.org/wiki/Independence_(probability_theory)),
i.e. when the occurrence of a component does not affect the probability of occurrence of the others,
the joint distribution is simply the product of the marginal distributions.

Joint distributions are defined using
[OTJointDistribution][gemseo.uncertainty.distribution.openturns.joint.OTJointDistribution]
or [SPJointDistribution][gemseo.uncertainty.distribution.scipy.joint.SPJointDistribution]
depending on the backend.

When the components are (stochastically) dependent,
the OpenTURNS backend supports [copulas](https://en.wikipedia.org/wiki/Copula_(statistics)),
which separate the marginal distributions from the dependence structure.
The dependence can be declared block-wise,
each copula covering a subset of the components,
the components outside these blocks being independent.
The way these blocks are declared conditions the cost
of the OpenTURNS reliability analyses,
which rely on the iso-probabilistic transformation of the joint distribution;
see [add_copula()][gemseo.space.random.RandomSpace.add_copula]
and [OTJointDistribution][gemseo.uncertainty.distribution.openturns.joint.OTJointDistribution].
Conditioning the components of each block-wise Rosenblatt transform
in ascending order,
as GEMSEO does,
coincides with the OpenTURNS iso-probabilistic transformation
[@Rosenblatt1952] [@Lebrun2009].

Finally,
the dependency structure can be visualized from the data
using a [pair plot][pair-plot] with the option `use_ranks` set to `True`.
This version of the pair plot is called a *copulogram* [@fekhari2024uncertainty].
As in a classic pair plot,
the lower part shows the joint distribution
by projecting the raw data onto planes defined by two components
while the diagonal represents the marginal distributions.
However,
the upper part is no longer symmetrical to the lower part
but shows the dependency structure between the components
by replacing raw data by their component-wise normalized ranks.
This empirical distribution tends to the copula
when the number of samples tends to infinity.

## Random space { #concept-random-space }

The [RandomSpace][gemseo.space.random.RandomSpace] models the uncertain inputs;
it declares random variables from the settings
of the marginal probability distributions of their components.
Each uncertain variable is added
using the [add_variable()][gemseo.space.random.RandomSpace.add_variable] method,
with one settings object per component,
and the dependency structure is defined using the [add_copula()][gemseo.space.random.RandomSpace.add_copula] method.
The read-only registry [variables][gemseo.space.random.RandomSpace.variables] then exposes
the joint distribution of the space (`variables.distribution`)
and the distribution of each random vector (`variables[name].distribution`),
used to generate samples for propagation and sensitivity analysis.
A random space has neither bounds setters, nor current value,
nor normalization, unlike a [DesignSpace][concept-design-space].

## Going further

!!! explanations
    - [Uncertainty propagation][concept-uncertainty-propagation]

!!! how-to
    - [Define probability distributions][probability-distributions]
    - [Fit a probability distribution from data][]
    - [Define a random space][]
