<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Surrogate modelling { #needs-surrogate-modelling }

## The problem { #needs-the-problem }

High-fidelity simulation models
can take minutes, hours, or even days to evaluate.
An optimization or DOE process
that requires hundreds or thousands of evaluations
quickly becomes prohibitively expensive.

Even when the simulation is affordable for a single run,
the cost multiplied by the number of evaluations
needed for optimization, sensitivity analysis, or uncertainty propagation
may be unacceptable.

## GEMSEO's answer: surrogate disciplines { #needs-gemseos-answer-surrogate-disciplines }

GEMSEO allows replacing a costly discipline
with a **surrogate discipline**---a cheap-to-evaluate approximation
built from a limited number of evaluations of the original discipline.

The workflow is straightforward:

1. **Sample** the original discipline using a design of experiments (DOE).
2. **Train** a regression model on the resulting input-output data.
3. **Substitute** the original discipline with the surrogate in the process.

Because the surrogate discipline
has the **same interface** (inputs and outputs) as the original,
the rest of the process is unaffected.
The MDO formulation, optimizer, and other disciplines
continue to work exactly as before,
but evaluations are now orders of magnitude faster.

GEMSEO provides a broad catalog of regression models
through its machine learning package:

- Linear and polynomial regression
- Radial basis functions (RBF)
- Gaussian process regression (Kriging)
- Neural networks
- Polynomial chaos expansion (PCE)
- And more, from scikit-learn and OpenTURNS

<!-- ![Machine learning taxonomy](../../assets/images/user_guide/ml_taxonomy.png){: style="display:block; margin:auto; max-width:70%" } -->

*GEMSEO's machine learning package covers regression, classification, and clustering.*

Quality measures (cross-validation, leave-one-out, etc.)
are available to assess the accuracy of the surrogate
before using it in the process.
