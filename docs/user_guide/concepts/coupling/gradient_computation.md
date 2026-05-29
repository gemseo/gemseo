---
description: "Gradient computation in coupled GEMSEO workflows: coupled adjoint theory, direct vs adjoint methods, differentiation modes, and Jacobian validation."
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

# Gradient computation { #concept-coupled-gradient-computation }

Gradient-based algorithms require the computation of the Jacobian matrix
of functions of interest, e.g. of the objective and constraints functions.
By definition, the Jacobian of the output vector
$\mathbf{f} = \left[f_1,\, \ldots, \, f_q \right]^\top$ with respect to
the input vector $\mathbf{x} = \left[x_1,\, \ldots, \, x_p \right]^\top$ is the matrix:

$$
\frac{\text{d}\mathbf{f}}{\text{d}\mathbf{x}} =
\begin{pmatrix}
    \frac{\text{d} f_1}{\text{d} x_1} & \ldots & \frac{\text{d} f_1}{\text{d} x_p}\\
    \vdots & \ddots & \vdots \\
    \frac{\text{d} f_q}{\text{d} x_1} & \ldots & \frac{\text{d} f_q}{\text{d} x_p}
\end{pmatrix}
$$

The Jacobian matrix may be either approximated
(with the finite difference or complex step methods),
or analytically computed.

The analytic computation of the derivatives is usually a better approach,
when it is affordable,
since it is cheaper and more precise than the approximations.

In the latter case, the complete MDO process has to be evaluated
for each perturbed point $(\mathbf{x}+h_j\mathbf{e}_j)$,
which scales badly with the number of design variables.

$$\frac{d f_i}{d x_j} =
\frac{f_i(\mathbf{x}+h_j\mathbf{e}_j)-f_i(\mathbf{x})}{h_j}+\mathcal{O}(h_j).$$

In case of the finite difference approximation,
tuning the step size is difficult since a tradeoff has to be found
between truncation error and cancellation error,
although some methods exist for that
([set_optimal_fd_step()][gemseo.core.discipline.discipline.Discipline.set_optimal_fd_step]).

As a result it is crucial to select an appropriate method to compute the gradient.

However, the computation of analytic derivatives is not straightforward.
It requires computing the Jacobian of the objective function and constraints,
assembled from a process (such as a chain or an MDA)
using the partial derivatives provided by all the disciplines.

In GEMSEO, for weakly coupled problems,
the generalized chain rule is used in reverse mode (from the outputs to the inputs)
based on [DisciplineChain][gemseo.core.chains.chain.DisciplineChain].

For strongly coupled problems,
when the process is based on a MDA [BaseMDA][gemseo.mda.base.BaseMDA],
an implicit differenciation approach is used,
with two variants, either direct or adjoint,
depending on the number of design variables
compared to the number of objectives and constraints.

The following section describes the coupled adjoint method.

## The coupled adjoint theory { #concept-the-coupled-adjoint-theory }

Consider two disciplines $\mathbf{F}_1$ and $\mathbf{F}_2$ defined such that:

$$
\left\{
\begin{aligned}
\mathbf{F}_1(\mathbf{x}, \mathbf{y}_2) = \mathbf{y}_1 \\
\mathbf{F}_2(\mathbf{x}, \mathbf{y}_1) = \mathbf{y}_2
\end{aligned}
\right.,
$$

where $\mathbf{y}_1$ and $\mathbf{y}_2$ are the strong coupling variables.
This system of non-linear equations can be rewritten

$$
\mathbf{R}(\mathbf{x}, \mathbf{y}) =
\left[
    \begin{aligned}
        \mathbf{F}_1(\mathbf{x}, \mathbf{y}_2) - \mathbf{y}_1 \\
        \mathbf{F}_2(\mathbf{x}, \mathbf{y}_1) - \mathbf{y}_2
    \end{aligned}
\right] = 0
\quad \text{where} \quad
\mathbf{y} =
\left[
    \begin{aligned}
        \mathbf{y}_1 \\
        \mathbf{y}_2
    \end{aligned}
\right].
$$

Performing an MDA on this set of disciplines consists in finding,
for a given value of $\mathbf{x}$,
the vector $\mathbf{y}^\star$ satisfying $\mathbf{R}(\mathbf{x}, \mathbf{y}^\star) = 0$.
When applicable,
the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem)
states that there exists a unique differentiable function $\varphi$ such that
$\mathbf{y} = \varphi(\mathbf{x}')$ and $\mathbf{R}(\mathbf{x}', \varphi(\mathbf{x}')) = 0$
for all $\mathbf{x}'$ in a neighbourhood of $\mathbf{x}$.
Differentiating this latter relation yields

$$
\frac{\text{d}\varphi}{\text{d}\mathbf{x}} = -
\frac{\partial\mathbf{R}}{\partial\mathbf{y}}^{-1}
\frac{\partial\mathbf{R}}{\partial\mathbf{x}}.
$$

Let us now consider a function
$\mathbf{f} : \mathbf{x}, \mathbf{y} \mapsto \mathbf{f}(\mathbf{x}, \mathbf{y})$
of interest (objective or constraints functions for instance).
Evaluating this function on points of the form
$\mathbf{x}', \varphi(\mathbf{x}')$
correspond to looking at the function values when the system of non-linear equations
is solved, that is at the equilibrium of the coupled system.
Let
$\widetilde{\mathbf{f}} = \mathbf{f}(\,\cdot\, , \varphi( \,\cdot\, ))$
denote this restriction,
applying the chain rule yields

$$
\frac{\text{d}\widetilde{\mathbf{f}}}{\text{d}\mathbf{x}} =
\frac{\partial\mathbf{f}}{\partial\mathbf{x}} +
\frac{\partial\mathbf{f}}{\partial\mathbf{y}}
\frac{\text{d}\varphi}{\text{d}\mathbf{x}}.
$$

Substituting $\text{d}\varphi / \text{d}\mathbf{x}$ for the expression obtained above,
we obtain the following expression for the total derivative:

$$
\frac{\text{d}\widetilde{\mathbf{f}}}{\text{d}\mathbf{x}} =
\frac{\partial\mathbf{f}}{\partial\mathbf{x}} -
\frac{\partial\mathbf{f}}{\partial\mathbf{y}}
\frac{\partial\mathbf{R}}{\partial\mathbf{y}}^{-1}
\frac{\partial\mathbf{R}}{\partial\mathbf{x}}.
$$

!!! note
    This expression involves only partial derivatives of the function of interest
    $\mathbf{f}$
    and of the residual $\mathbf{R}$,
    that is of $\mathbf{F}_1$ and $\mathbf{F}_2$.
    In particular,
    the function $\varphi$ from the implicit function theorem no longer appears.

## Adjoint versus direct methods { #concept-adjoint-versus-direct-methods }

The computing of the total derivatives requires to have all the partial derivatives
involved in the different terms, but also to solve linear systems involving the
operator $\partial\mathbf{R} / \partial\mathbf{y}$, which actually drives the
computational cost.

Two approaches are possible to compute the previous equation:

- **Direct mode.** Let us consider $\mathbf{x} = \left[x_1, \, \ldots, \, x_p\right]$,
    then one first needs to solve the $p$ following linear systems:

    $$
    \frac{\partial\mathbf{R}}{\partial\mathbf{y}} \mathbf{z}_i =
    \frac{\partial\mathbf{R}}{\partial x_i}.
    $$

    Denoting $\mathbf{Z} = \left[\mathbf{z}_1, \, \ldots, \, \mathbf{z}_p\right]$,
    the total derivatives are obtained via

    $$
    \frac{\text{d}\widetilde{\mathbf{f}}}{\text{d}\mathbf{x}} =
    \frac{\partial\mathbf{f}}{\partial\mathbf{x}} -
    \frac{\partial\mathbf{f}}{\partial\mathbf{y}} \mathbf{Z}.
    $$

- **Adjoint mode.** Let us consider $\mathbf{f} = \left[f_1, \, \ldots, \, f_q\right]$,
    then one can alternatively begin with solving the $q$ linear systems:

    $$
    \frac{\partial\mathbf{R}}{\partial\mathbf{y}}^\top \mathbf{w}_i =
    \frac{\partial f_i }{\partial \mathbf{y}}^\top.
    $$

    Denoting $\mathbf{W} = \left[\mathbf{w}_1, \, \ldots, \, \mathbf{w}_q\right]$,
    the total derivatives are obtained via

    $$
    \frac{\text{d}\widetilde{\mathbf{f}}}{\text{d}\mathbf{x}} =
    \frac{\partial\mathbf{f}}{\partial\mathbf{x}} -
    \mathbf{W}^\top \frac{\partial\mathbf{R}}{\partial\mathbf{x}}.
    $$

The direct method requires to solve $p$ linear systems
(one per design variable component) involving $\partial\mathbf{R} / \partial\mathbf{y}$,
while the adjoint method requires to solve $q$ linear systems
(one per function of interest component) involving
$\partial\mathbf{R} / \partial\mathbf{y}^\top$. In particular the adjoint method requires
to have access to the adjoint of the partial derivatives.

Both the direct and adjoint methods are implemented in GEMSEO.
The switch between the direct and adjoint method is automatic and depends on $p$ and $q$,
but it can also be forced by the user.

## Differentiation methods { #concept-differentiation-methods }

Differentiation can be performed at either [Discipline][gemseo.core.discipline.discipline.Discipline] or workflow level.

### Discipline differentiation { #concept-discipline-differentiation }

Each [Discipline][gemseo.core.discipline.discipline.Discipline]
can compute its Jacobian matrix in different ways.
See [Jacobian][concept-discipline-jacobian] for details.

GEMSEO then assembles the total coupled Jacobian automatically
using the direct or adjoint method
(see [Adjoint versus direct methods][concept-adjoint-versus-direct-methods]).

!!! note
    It is possible to combine different types of differentiation methods.
    Some disciplines may compute analytic Jacobians, while others don't.
    In that case, GEMSEO is able to combine the different Jacobians.

!!! note
    For analytic differentiation,
    the switch between direct and adjoint is automatic by default,
    driven by the ratio of outputs to design variables.
    It can be overridden by setting
    [Discipline.linearization_mode][gemseo.core.discipline.discipline.Discipline.linearization_mode]
    to `"direct"` or `"adjoint"` on individual disciplines.

### The workflow

It is also possible to get the overall Jacobian matrix or operator on the workflow level.
In that case, the differentiation method is selected at the
[EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario] level via
[EvaluationScenario.set_differentiation_method()][gemseo.scenarios.evaluation.EvaluationScenario.set_differentiation_method]:

- **Finite differences** (default):
  each output is numerically perturbed by a small step $h_j$ per design variable.
  First-order accurate in $h_j$.
  The entire MDO process must be re-evaluated for each perturbed point,
  so cost scales linearly with the number of design variables.

- **Complex step**:
  a complex perturbation $ih_j$ avoids cancellation errors,
  giving machine-precision accuracy without reducing $h_j$ to unsafe values.
  Each discipline must support complex-valued inputs and outputs.
