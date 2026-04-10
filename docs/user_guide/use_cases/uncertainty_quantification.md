<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Uncertainty quantification { #usecases-uncertainty-quantification }

## What it means { #usecases-what-it-means }

In reality,
input parameters are never known with perfect certainty:
material properties vary from one specimen to another,
operating conditions fluctuate,
and models themselves are imperfect approximations.
**Uncertainty quantification** (UQ) addresses
the question of how these sources of uncertainty
affect the outputs of interest.

## Why it matters { #usecases-why-it-matters }

Ignoring uncertainty can lead to designs
that look optimal in simulation
but fail under real-world conditions.
UQ provides the tools to:

- **Represent** sources of uncertainty as probability distributions, whether from data or expert opinion.
- **Propagate** uncertainties through a model or a coupled system to assess the resulting output variability.
- **Quantify** output uncertainty through statistics (mean, variance, quantiles) or probability distributions.
- **Analyze sensitivity**: identify which uncertain inputs contribute most to output variability, and which can be neglected.
- **Assess reliability**: estimate the probability that a quantity of interest exceeds a given threshold.
- **Optimize under uncertainty**: find designs that are robust to input variability, for instance by maximizing an average performance while guaranteeing constraints in 99% of cases.

## How GEMSEO does it { #usecases-how-gemseo-does-it }

GEMSEO implements UQ through a dedicated package
that integrates with its MDO capabilities.

### Uncertain variables { #usecases-uncertain-variables }

GEMSEO extends the notion of **design space**
with a `ParameterSpace` that defines
both deterministic and uncertain variables.
Uncertain variables are described by probability distributions
(normal, uniform, triangular, etc.),
which can be fitted from data
or specified by the user.

### Uncertainty propagation { #usecases-uncertainty-propagation }

Propagation is performed by sampling the uncertain inputs
using DOE algorithms (Monte Carlo, LHS, etc.)
and evaluating the disciplines at each sample.
The resulting output samples provide empirical distributions
for the quantities of interest.

Because GEMSEO's UQ features use the same discipline and scenario framework
as optimization and DOE,
they are **interoperable with all MDO features**.
For instance, one can propagate uncertainties
through a coupled multidisciplinary system
by combining an MDA with a sampling strategy.

### Sensitivity analysis { #usecases-sensitivity-analysis }

Sensitivity analysis identifies
which uncertain inputs most influence the outputs.
GEMSEO provides several methods:

- Sobol indices (variance-based, via OpenTURNS)
- Morris screening (one-at-a-time)
- Correlation-based methods

### Towards robust MDO { #usecases-towards-robust-mdo }

The UQ capabilities are a building block
for robust and reliable MDO:
combining optimization with uncertainty management
to find designs that perform well
not just at a nominal point,
but across a range of uncertain conditions.
