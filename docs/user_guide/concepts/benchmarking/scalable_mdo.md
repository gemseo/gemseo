---
description: "Scalable MDO problems bundled with GEMSEO: linear, quadratic parametric and data-driven approaches for studying algorithmic scalability."
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

# Scalable MDO problems { #concept-scalable-mdo }

A *scalable MDO problem* is a synthetic MDO problem
whose number of disciplines, local variables, shared variables and coupling variables
can all be set independently.
This makes it possible to study how the cost or quality of an algorithm
depends on problem size without changing the underlying problem structure.

GEMSEO provides three scalable MDO approaches
via the [gemseo.problems.mdo.scalable][gemseo.problems.mdo.scalable] package.

## Linear scalable MDO { #concept-linear-scalable-mdo }

### Purpose { #concept-linear-purpose }

The linear scalable MDO generates $N$ coupled disciplines
with controllable input and output sizes.
It is designed to stress-test MDO formulations and MDA solvers
at arbitrary problem dimensions while keeping disciplines cheap to evaluate.

### Properties { #concept-linear-properties }

Each discipline computes a linear mapping

$$y = A\,x + b$$

where $A$ and $b$ are random matrices and vectors drawn at construction time.
Coupling is achieved by feeding outputs of one discipline as inputs to another.
By construction, the MDA of the generated disciplines always converges
when the inputs are in $[0, 1]$.
The analytic Jacobian $\partial y/\partial x = A$ is available for gradient-based methods.

??? abstract "API"

    - [LinearDiscipline][gemseo.problems.mdo.scalable.linear.linear_discipline.LinearDiscipline]:
      a single linear discipline.
    - [create_disciplines_from_sizes][gemseo.problems.mdo.scalable.linear.disciplines_generator.create_disciplines_from_sizes]:
      and
      [create_disciplines_from_desc][gemseo.problems.mdo.scalable.linear.disciplines_generator.create_disciplines_from_desc]
      generates a set of coupled linear disciplines from a coupling specification.

## Quadratic scalable MDO { #concept-quadratic-scalable-mdo }

### Purpose { #concept-quadratic-purpose }

The quadratic parametric scalable MDO builds a set of strongly coupled disciplines
with quadratic coupling equations and a known analytical optimum.
It enables verification of MDO solvers: the optimizer's solution
can be checked against the exact optimum.

### Properties { #concept-quadratic-properties }

The problem consists of $N$ scalable disciplines plus a main discipline
computing the objective and constraints.
All disciplines are defined on a unit design space, $x \in [0, 1]$.
The number of disciplines $N$, the size of local variables per discipline,
the size of shared variables, and the coupling variable sizes are all configurable.
Because the optimum is known analytically,
the problem is well-suited for convergence studies.

??? abstract "API"

    - [ScalableProblem][gemseo.problems.mdo.scalable.parametric.scalable_problem.ScalableProblem]:
      assembles disciplines, design space and analytical solution.
    - [ScalableDesignSpace][gemseo.problems.mdo.scalable.parametric.scalable_design_space.ScalableDesignSpace]
    - [ScalableDiscipline][gemseo.problems.mdo.scalable.parametric.disciplines.scalable_discipline.ScalableDiscipline]
    - [MainDiscipline][gemseo.problems.mdo.scalable.parametric.disciplines.main_discipline.MainDiscipline]

## Data-driven scalable MDO { #concept-data-driven-scalable-mdo }

### Methodology { #concept-methodology }

The data-driven approach builds scalable disciplines from real input-output training data.
Given a dataset for each discipline and a target problem dimension,
it constructs surrogate disciplines that generalize the original behaviour
to the new dimension.

#### One-dimensional restriction { #concept-one-dimensional-restriction }

Each high-dimensional input-output relationship is first projected onto
a one-dimensional latent variable using a dimensionality reduction technique
(e.g. partial least squares or principal component analysis).
This yields a low-complexity representation of the original function
that can be fitted reliably even with small datasets.

#### Interpolation { #concept-interpolation }

A surrogate model (e.g. a radial basis function network)
is fitted to the one-dimensional restriction.
The surrogate captures the essential nonlinear structure of the original discipline
along the dominant mode of variation.

#### Input-output dependency { #concept-input-output-dependency }

Not all inputs affect all outputs.
An input-output dependency analysis is performed to identify which input variables
influence which output variables.
This produces a sparse coupling graph that avoids introducing spurious dependencies
in the scaled problem.

#### Extrapolation { #concept-extrapolation }

Once the surrogate is fitted in the original dimension,
it is extended to the target dimension by applying a variable transformation
that maps the new inputs and outputs back to the one-dimensional latent space.
This allows the scaled discipline to be evaluated at any target size.

### Properties { #concept-data-driven-properties }

The data-driven approach preserves the qualitative behaviour of real disciplinary data
while allowing arbitrary scaling of variable sizes.
It is particularly useful when one wants to study algorithmic scalability
starting from a real engineering problem rather than a purely synthetic one.

??? abstract "API"

    - [ScalableProblem][gemseo.problems.mdo.scalable.data_driven.problem.ScalableProblem]:
      orchestrates the full workflow from dataset to MDO scenario.
    - [DataDrivenScalableDiscipline][gemseo.problems.mdo.scalable.data_driven.discipline.DataDrivenScalableDiscipline]:
      a single scalable discipline built from a training dataset.
    - [ScalableModel][gemseo.problems.mdo.scalable.data_driven.model.ScalableModel]:
      base class for the underlying surrogate model;
      subclass it to plug in a custom interpolation method.
