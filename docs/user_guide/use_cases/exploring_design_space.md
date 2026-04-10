<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Exploring design space { #usecases-exploring-design-space }

## What it means { #usecases-what-it-means }

Before optimizing,
engineers often need to **explore** the design space
to understand how the outputs of a system
depend on its inputs.
A **Design of Experiments** (DOE)
is a systematic method for selecting input values
and evaluating the corresponding outputs.

## Why it matters { #usecases-why-it-matters }

A DOE serves several purposes:

- **Understanding**: visualize trends, identify dominant inputs, and detect nonlinear or unexpected behaviors.
- **Sensitivity analysis**: determine which inputs have the most influence on the outputs.
- **Surrogate modelling**: generate the training data needed to build a cheap approximation of a costly discipline.
- **Trade-off analysis**: compare configurations across the design space without committing to a single objective.

A DOE provides a broad view of the design space,
whereas optimization focuses on a single point.
Both are complementary:
a DOE can inform the choice of an optimizer
and highlight regions of interest.

## How GEMSEO does it { #usecases-how-gemseo-does-it }

In GEMSEO, a DOE is executed through a **scenario**:
the same framework used for optimization.
The user defines:

1. The **disciplines** to evaluate.
2. The **design space**: the input variables and their bounds.
3. The **DOE algorithm** and its settings (e.g. number of samples).

GEMSEO then generates the samples,
evaluates the disciplines at each sample point,
and stores the results for post-processing.

### Available DOE algorithms { #usecases-available-doe-algorithms }

GEMSEO wraps DOE algorithms from PyDOE and OpenTURNS:

- **Monte Carlo**: random uniform sampling.
- **Latin Hypercube Sampling (LHS)**: ensures each input range is evenly covered. Optimized variants are available.
- **Low-discrepancy sequences**: Sobol, Halton, Faure---designed to cover the space as uniformly as possible.
- **Stratified DOEs**: full factorial, central composite, Box-Behnken, Plackett-Burman---make inputs vary by discrete levels.
- **Custom DOE**: the user provides their own sample points as a NumPy array or CSV file.

### Post-processing { #usecases-post-processing }

The evaluation results are stored as a **Dataset**
(a multi-index pandas DataFrame)
and can be visualized through GEMSEO's post-processing tools:
scatter plots, parallel coordinates, surface plots, correlation matrices, etc.

Quality measures
($\varphi_p$ criterion, minimum distance, discrepancy)
are available to assess and compare DOE designs.
