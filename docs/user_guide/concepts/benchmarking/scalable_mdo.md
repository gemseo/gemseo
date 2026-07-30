---
reading_time: true
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

### Purpose { #concept-data-driven-purpose }

The data-driven approach, proposed in [@Vanaret2017],
builds scalable disciplines from the input-output data of an original MDO problem.
Given a dataset for each discipline and a target problem dimension,
it constructs cheap disciplines generalizing the original behaviour to that dimension.

It allows to choose an [MDO formulation][concept-mdo-formulations]:

- for the original problem from which the scalable problem derives, or
- for a family of problems having a greater number of design and coupling variables
  and common properties with the original problem.

According to the authors, these scalable problems
*"preserve the functional characteristics of the original problem,
and they proved useful in performing a rapid benchmarking of MDO formulations"*.
This *"provides insights on the scalability of MDO architectures
with respect to the dimensions of the problem.
This may be achieved without having to execute the MDO processes with the original models.
Our methodology thus requires a limited number of evaluations of the original models
that is independent of the desired dimensions of the design
and the coupling variables of the scalable problem."*

!!! info "See also"

    The scalable model is illustrated in [several examples][scalable-model].

### Methodology { #concept-methodology }

The methodology

1. builds a surrogate model $\Phi^{(int)}$
   for each discipline $\Phi$ of the original problem
   with a limited amount of evaluations $T$,
2. extrapolates the surrogate model $\Phi^{(ext)}$ to an arbitrary dimension.

It preserves the interface of the original problem,
namely the names of the inputs (design variables)
and the outputs (coupling and state variables).
Any high-fidelity discipline of the original problem
may therefore be replaced by a cheap scalable discipline generated by the methodology,
with the [properties][concept-data-driven-properties] listed below.

#### One-dimensional restriction { #concept-one-dimensional-restriction }

The original model $\Phi:\mathbb{R}^n\rightarrow\mathbb{R}^m$
is restricted to a one-dimensional function $\Phi^{(1d)}:[0,1]\rightarrow\mathbb{R}^m$
by evaluating it along a diagonal line
in the domain $[x_1,\overline{x_1}]\times\ldots\times[x_n,\overline{x_n}]$:

$$\Phi^{(1d)}(t)=\Phi\left(x_1+t(\overline{x_1}-x_1),\ldots,x_n+t(\overline{x_n}-x_n)\right)$$

#### Interpolation { #concept-interpolation }

For any component $i\in\{1,\ldots,m\}$ of $\Phi^{(1d)}$,
the direct image of $T$, a finite subset of $[0,1]$ with cardinality $|T|$, is

$$\Phi_i^{(1d)}(T) = \left\{\Phi_i^{(1d)}(t)\,|\,t\in T\right\}$$

mapping from $[0,1]$ to $[m_i, M_i]$
where $m_i$ and $M_i$ are respectively the minimal and maximal values
reached by $\Phi_i^{(1d)}$ over $T$.

The scaled version of $\Phi_i^{(1d)}(T)$ is

$$\Phi_i^{(s1d)}(T) = \left\{\left.\frac{\Phi_i^{(1d)}(t)-m_i}{M_i-m_i}\,\right|\,t\in T\right\}$$

mapping from $[0,1]$ to $[0,1]$.

Then, each component $i$ of $\Phi^{(1d)}(t)$ is approximated
by a polynomial interpolation $\Phi_i^{(int)}$
over the data $\left(T,\Phi_i^{(s1d)}(T)\right)$.

#### Input-output dependency { #concept-input-output-dependency }

Dependencies between inputs and outputs
can be represented by a sparse dependency matrix $S$ where

- each block row represents a function of the problem (constraint or coupling),
- each block column represents an input (design variable or coupling),
- a nonzero element represents the dependency of a particular component of a function
  with respect to a particular component of an input.

In practice, these dependencies are not precisely known.
Consequently, the matrix $S$ is randomly computed by block
by means of a density factor
(the filling of a block is proportional to this density factor).

Furthermore, initially taken in $\mathcal{M}_{n,m}(\mathbb{R})$,
this matrix $S$ can be taken in $\mathcal{M}_{n_x,n_y}(\mathbb{R})$
where the number of inputs $n_x$ and the number of outputs $n_y$
of the scalable model are freely chosen by the user.

#### Extrapolation { #concept-extrapolation }

Once $n_x$ and $n_y$ are chosen,
we build the function $\Phi^{(ext)}:[0,1]^{n_x}\rightarrow[0,1]^{n_y}$
extrapolating $\Phi^{(int)}:[0,1]\rightarrow[0,1]^{m}$ to $n_y$ dimensions:

$$\Phi_i^{(ext)}(x)=\frac{1}{|S_{i.}|}\sum_{j\in S_{i.}} \Phi_{k_i}^{(int)}(x_j)$$

where

- $S_{i.}$ represents the nonzero elements of the $i$-th row of the dependency matrix $S$,
- $k_i$ is a uniform random variable over $\left\{1,\ldots,m\right\}$.

### Properties { #concept-data-driven-properties }

The data-driven approach preserves the qualitative behaviour of real disciplinary data
while allowing arbitrary scaling of variable sizes.
It is particularly useful when one wants to study algorithmic scalability
starting from a real engineering problem rather than a purely synthetic one.

The methodology guarantees the following strong properties:

- **Existence of a solution to the coupling problem**.
  An equilibrium between all disciplines exists
  for any value of the design variables $x$.
- **Preservation of ratio**.
  When $n_y$ approaches $+\infty$,
  the ratio of components of the original functions is preserved.
- **Existence of a minimum**.
  There exists a feasible solution to the scalable problem,
  for any dimension of inputs and outputs.
- **Existence of derivatives**.
  The scalable extrapolations are continuously differentiable
  with respect to their inputs.
- **Existence of bounds on the target coupling variables**.
  All inputs and outputs belong to $[0, 1]$,
  which ensures that all optimization variables are bounded,
  in particular coupling variables in IDF.

??? abstract "API"

    - [ScalableProblem][gemseo.problems.mdo.scalable.data_driven.problem.ScalableProblem]:
      orchestrates the full workflow from dataset to MDO scenario.
    - [DataDrivenScalableDiscipline][gemseo.problems.mdo.scalable.data_driven.discipline.DataDrivenScalableDiscipline]:
      a single scalable discipline built from a training dataset.
    - [ScalableModel][gemseo.problems.mdo.scalable.data_driven.model.ScalableModel]:
      base class for the underlying surrogate model;
      subclass it to plug in a custom interpolation method.
