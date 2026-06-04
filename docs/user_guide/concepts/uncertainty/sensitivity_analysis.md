---
description: "Sensitivity analysis quantifies the relative contribution of each uncertain input to the variability of the model outputs."
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

# Sensitivity analysis { #concept-sensitivity-analysis }

After [propagating uncertainty][concept-uncertainty-propagation] through a model,
a natural question arises: *which uncertain inputs are responsible for the output variability?*
Sensitivity analysis answers this by computing **sensitivity indices**,
i.e. scalar scores that rank the uncertain inputs by their influence on each output.

Global sensitivity analysis (GSA) evaluates the output variation across the entire input distribution,
capturing nonlinear effects and interactions between inputs.
Its results depend on the chosen input distributions, not on a fixed reference point,
making it more informative than local (gradient-based) approaches
for models that are nonlinear or whose inputs are uncertain over a wide range.

GEMSEO provides four sensitivity analysis techniques.

## Available sensitivity analyses { #concept-sensitivity-analyses }

| Analysis    | Class                                                                                 | Approach            | Key indices                          |
|-------------|---------------------------------------------------------------------------------------|---------------------|--------------------------------------|
| Correlation | [CorrelationAnalysis][gemseo.uncertainty.sensitivity.correlation.CorrelationAnalysis] | Linear / Monotonous | Pearson, Spearman, PCC, PRCC         |
| Morris      | [MorrisAnalysis][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis]                | Screening           | $\mu^*$, $\sigma$                    |
| Sobol'      | [SobolAnalysis][gemseo.uncertainty.sensitivity.sobol.SobolAnalysis]                   | Variance-based      | First-order $S_1$, total-order $S_T$ |
| HSIC        | [HSICAnalysis][gemseo.uncertainty.sensitivity.hsic.HSICAnalysis]                      | Kernel-based        | HSIC, $R^2$-HSIC                     |

Correlation, Sobol' and HSIC analyses integrate features from [OpenTURNS](https://openturns.github.io/www/)
while Morris analysis is a custom implementation.

### Correlation analysis

Correlation analysis measures the linear (Pearson) or rank-based (Spearman) association
between each input and each output.
Partial correlation coefficients (PCC, PRCC) remove the effect of the other inputs,
isolating the direct contribution of each one.
This method is cheap but limited to linear or monotonic relationships.

### Morris analysis

Morris analysis is a screening method designed for high-dimensional problems.
It repeats a one-at-a-time (OAT) perturbation scheme $R$ times and estimates for each input:
$\mu_i^*$ — the mean absolute elementary effect, measuring the overall degree of influence —
and $\sigma_i$ — its standard deviation, which reveals nonlinearity and interaction with other inputs.
Inputs with small $\mu_i^*$ are non-influential and can be fixed;
inputs with large $\sigma_i$ relative to $\mu_i^*$ interact strongly with others.
It is efficient for initial screening but does not quantify the exact share of variance.

### Sobol' analysis

Sobol' analysis decomposes the total output variance by the [Sobol'-Hoeffding identity](https://openturns.github.io/openturns/latest/theory/reliability_sensitivity/sensitivity_sobol.html):

$$\mathbb{V}[Y] = \sum_i V_i + \sum_{i<j} V_{ij} + \cdots$$

where $V_i = \mathbb{V}[\mathbb{E}[Y|X_i]]$ is the variance explained by $X_i$ alone.
The first-order index $S_i = V_i / \mathbb{V}[Y]$ measures the direct effect of $X_i$,
while the total-order index $S_{T_i}$ sums all terms involving $X_i$ (direct and interactions).
Both indices lie in $[0,1]$; a large gap $S_{T_i} - S_i$ signals strong interaction effects.
It is the most informative method but requires a large number of model evaluations.

### HSIC analysis

HSIC analysis uses the Hilbert–Schmidt independence criterion (HSIC),
a kernel-based statistical dependence measure that detects any type of relationship,
including nonlinear and non-monotonic ones.
In addition to GSA (default),
GEMSEO can use HSIC indices
to identify the input variables
most likely to cause the output to reach a certain domain (target sensitivity analysis)
or most likely to cause the output to deviate from a nominal value
under the condition that the considered output is in a certain domain (conditional sensitivity analysis).

## Common interface { #concept-sensitivity-interface }

All sensitivity analyses follow the same workflow:

1. **Compute samples** — [compute_samples()][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.compute_samples] draws input–output samples
   from the [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace].
   Alternatively, an existing [IODataset][gemseo.datasets.io_dataset.IODataset] can be reused directly.
2. **Compute indices** — [compute_indices()][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.compute_indices]
   derives the sensitivity indices from the samples.
3. **Visualize** — methods such as [plot_bar()][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.plot_bar] and [plot_radar()][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.plot_radar] display the indices.
4. **Export** — [to_dataset()][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.to_dataset]
   exports the indices as a [Dataset][concept-dataset].

Indices can be standardized so that different methods are compared on the same scale
(e.g. using [plot_comparison()][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.plot_comparison]),
and inputs can be sorted by decreasing influence via [sort_input_variables()][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis.sort_input_variables].

## Going further

!!! explanations
    - [Uncertainty propagation][concept-uncertainty-propagation]
