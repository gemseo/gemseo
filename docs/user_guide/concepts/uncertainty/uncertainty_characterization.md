---
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
The integrate (or sum in the case of a discrete variable) of all values is 1.
For example,
when rolling an unloaded die,
the possible outcomes are 1, 2, ..., 6 and each outcome has a probability of occurrence equal to 1/6.

GEMSEO allows you to characterize uncertain scalar variables as random variables,
using a univariate distributions,
and uncertain vector variables as random vectors,
using joint distributions.

## Univariate distributions

GEMSEO has a [BaseUnivariateDistribution][gemseo.uncertainty.distributions.base_univariate.BaseUnivariateDistribution] class
to wrap univariate distributions from external libraries.
The interface offers many services:
querying the probability density function (PDF) and cumulative distribution function (CDF),
plotting the PDF and CDF,
sampling from the distribution and
computing statistical moments (mean, standard deviation).

The available backends are
[OpenTURNS](https://openturns.github.io/www/) and [SciPy](https://docs.scipy.org/doc/scipy)
with the associated classes
[OTDistribution][gemseo.uncertainty.distributions.openturns.distribution.OTDistribution]
and [SPDistribution][gemseo.uncertainty.distributions.scipy.distribution.SPDistribution].
Any distribution available in these external libraries can be created from these classes.
Subclasses also exist for classical distributions,
such as
[OTUniformDistribution][gemseo.uncertainty.distributions.openturns.uniform.OTUniformDistribution],
[OTNormalDistribution][gemseo.uncertainty.distributions.openturns.normal.OTNormalDistribution]
and
[OTTriangularDistribution][gemseo.uncertainty.distributions.openturns.triangular.OTTriangularDistribution]
for OpenTURNS,
and
[SPUniformDistribution][gemseo.uncertainty.distributions.scipy.uniform.SPUniformDistribution],
[SPNormalDistribution][gemseo.uncertainty.distributions.scipy.normal.SPNormalDistribution]
and
[SPTriangularDistribution][gemseo.uncertainty.distributions.scipy.triangular.SPTriangularDistribution]
for SciPy.

These distributions are defined from settings,
e.g. [OTUniformDistribution_Settings][gemseo.uncertainty.distributions.openturns.uniform_settings.OTUniformDistribution_Settings]
in the case of the [OTUniformDistribution][gemseo.uncertainty.distributions.openturns.uniform.OTUniformDistribution].

## Choosing a distribution { #concept-choosing-a-distribution }

The choice of distribution depends on whether data are available.

### Without data { #concept-choosing-a-distribution-without-data }

Without data, expert knowledge can be translated into a distribution
by the [maximum entropy principle](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy):
among all distributions consistent with the available information,
choose the one with the highest Shannon entropy,
i.e., the least informative one.

Common mappings are:

| Prior knowledge                            | Distribution                          |
|--------------------------------------------|---------------------------------------|
| Only bounds $[a, b]$ known                 | Uniform $\mathcal{U}(a, b)$           |
| Bounds and most likely value $c$           | Triangular $\mathcal{T}(a, b, c)$     |
| Standard deviation $\sigma$ and mean $\mu$ | Normal $\mathcal{N}(\mu, \sigma^2)$   |
| Strictly positive, mean $\mu$              | Exponential with parameter $\mu^{-1}$ |

The user simply needs to provide this prior knowledge to the relevant distribution,
e.g. `OTUniformDistribution_Settings(minimum=a, maximum=b)`.

### With data { #concept-choosing-a-distribution-with-data }

With data,
the parameters of a distribution can be estimated by a statistical technique,
such as the [maximum likelihood method](https://en.wikipedia.org/wiki/Method_of_moments_(statistics))
or the [method of moments](https://en.wikipedia.org/wiki/Method_of_moments_(statistics)).

[OTDistributionFitter][gemseo.uncertainty.distributions.openturns.distribution_fitter.OTDistributionFitter]
automatically selects the best-fitting distribution from a set of candidates
using this kind of technique.
It tests each candidate using one of these goodness-of-fit criteria:
the [Bayesian information criterion (BIC)](https://en.wikipedia.org/wiki/Bayesian_information_criterion),
the [Kolmogorov–Smirnov statistical hypothesis test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
or the [Chi-squared statistical hypothesis test](https://en.wikipedia.org/wiki/Chi-squared_test))
and returns the distribution that best matches the data.

In the case of SciPy,
[SPDistributionFitter][gemseo.uncertainty.distributions.scipy.distribution_fitter.SPDistributionFitter]
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
called marginal marginal distributions.
When the components are [independent](https://en.wikipedia.org/wiki/Independence_(probability_theory)),
i.e. when the occurrence of a component does not affect the probability of occurrence of the others,
the joint distribution is simply the product of the marginal distributions.

Joint distributions are defined using
[OTJointDistribution][gemseo.uncertainty.distributions.openturns.joint.OTJointDistribution]
or [SPJointDistribution][gemseo.uncertainty.distributions.scipy.joint.SPJointDistribution]
depending on the backend.

When the components are (stochastically) dependent,
the OpenTURNS backend supports [copulas](https://en.wikipedia.org/wiki/Copula_(statistics)),
which separate the marginal distributions from the dependence structure.

Finally,
the dependency structure can be visualized from the data
using a [pair plot][pair-plot] with the option `use_ranks` set to `True`.
This version of the pair plot is called a *copulogram*[^1].
As in a classic pair plot,
the lower part shows the joint distribution
by projecting the raw data onto planes defined by two components
while the diagonal represents the marginal distributions.
However,
the upper part is no longer symmetrical to the lower part
but shows the dependency structure between the components
by replacing raw data by their component-wise normalized ranks.
This empirical distribution tends to the the copula
when the number of samples tends to infinity.

[^1]: Elias Fekhari. Uncertainty quantification in multi-physics model for wind turbine asset management. Signal and Image processing. Université Côte d'Azur, 2024. English. [⟨NNT : 2024COAZ4006⟩](https://www.theses.fr/2024COAZ4006). [⟨tel-04617148⟩](https://theses.hal.science/tel-04617148v1).

## Parameter space { #concept-parameter-space }

The [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
extends the [DesignSpace][concept-design-space]
to declare uncertain variables alongside deterministic ones.
Each uncertain variable is added with its marginal distribution
using either the [add_random_variable()][gemseo.algos.parameter_space.ParameterSpace.add_random_variable] or the [add_random_vector()][gemseo.algos.parameter_space.ParameterSpace.add_random_vector] method,
and the dependency structure is defined using the [add_copula()][gemseo.algos.parameter_space.ParameterSpace.add_copula] method.
The [distribution][gemseo.algos.parameter_space.ParameterSpace.distribution] attribute then exposes the corresponding joint distribution
used to generate samples for propagation and sensitivity analysis.

## Going further

!!! explanations
    - [Uncertainty propagation][concept-uncertainty-propagation]
