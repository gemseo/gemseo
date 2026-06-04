---
description: "Uncertainty propagation evaluates how uncertain inputs affect model outputs by sampling their joint distribution and computing output statistics."
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

# Uncertainty propagation { #concept-uncertainty-propagation }

Once uncertain inputs are [characterized][concept-uncertainty-characterization],
we are trying to calculate the induced output uncertainties.
This is the purpose of uncertainty propagation.

GEMSEO addresses this via sampling the multidisciplinary system:
the [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
generates samples from the joint input distribution,
an [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario]
evaluates the model at each sample,
and statistics are computed on the resulting output [IODataset][gemseo.datasets.io_dataset.IODataset].

## Sampling { #concept-sampling-uncertain-inputs }

Input samples are drawn from the joint probability distribution defined in the [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
using a design of experiment (DOE) algorithm — typically crude MC or a space-filling variant
such as [Latin hypercube sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) (LHS).
In practice,
we create an [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario]
and execute it using a DOE algorithm, e.g. MC.
Behind the scenes,
the latter samples the inputs
and evaluates the multidisciplinary process resulting from the formulation at every sampled input point,
The result is a dataset of input–output pairs,
which is the raw material for both [statistics approximation][concept-output-statistics]
and [sensitivity analysis][concept-sensitivity-analysis].

Then,
the statistics can be estimated empirically,
e.g. $\frac{1}{N}\sum_{i=1}^N Y^{(i)}$ is the estimator of the mean $\mathbb{E}[Y]$.

However,
the convergence rate of MC estimation is $O(N^{-1/2})$, which is relatively slow:
halving the estimation error requires four times as many samples.
However,
this rate is independent of the input dimension,
which makes MC robust in high-dimensional settings.

Space-filling DOE strategies improve on pure random sampling.
[Latin hypercube sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) (LHS)
stratifies each input into $N$ iso-probabilistic intervals,
ensuring that every stratum is represented exactly once,
which can increase the convergence rate compared to crude MC.
Quasi-MC methods,
a.k.a. [low-discrepancy sequences](https://en.wikipedia.org/wiki/Low-discrepancy_sequence),
such as Sobol', Halton, or Faure,
achieve an even faster convergence rate of $O((\log(N))^d/N)$ at the cost of being sensitive to the input dimension $d$.

## Output statistics { #concept-output-statistics }

The function [create_statistics()][gemseo.uncertainty.create_statistics]
builds a statistics toolbox from the output dataset.
For a model output $Y = f(X)$, typical quantities of interest include:
the mean $\mathbb{E}[Y]$, the variance $\mathbb{V}[Y]$, the standard deviation $\sqrt{\mathbb{V}[Y]}$,
a quantile $q_\alpha$ such that $\mathbb{P}[Y \leq q_\alpha] = \alpha$,
and the probability of exceeding a threshold $y_0$, i.e., $\mathbb{P}[Y > y_0]$.

Two modes are available:

- [EmpiricalStatistics][gemseo.uncertainty.statistics.empirical.EmpiricalStatistics]:
  compute these quantities of interest directly from the samples,
  without assuming any parametric form for the output distribution.
  Empirical statistics require no modelling assumptions
  but their accuracy depends entirely on the sample size and space-fillingness.
- [OTParametricStatistics][gemseo.uncertainty.statistics.ot_parametric.OTParametricStatistics]
  and [SPParametricStatistics][gemseo.uncertainty.statistics.sp_parametric.SPParametricStatistics]:
  fit a probability distribution to the output samples
  and derive the statistics analytically from the fitted distribution.
  The best distribution is selected from a set of candidates
  using a goodness-of-fit criterion,
  exactly as in [distribution fitting for inputs][concept-choosing-a-distribution-with-data].
  Parametric statistics can be more accurate for small sample sizes
  when the assumed family of distributions is correct,
  but they introduce a modelling error if it is not.

## Going further

!!! explanations
    - [Uncertainty characterization][concept-uncertainty-characterization]
    - [Sensitivity analysis][concept-sensitivity-analysis]

!!! how-to
    - [Execute a scenario][execute-a-scenario]
