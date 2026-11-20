---
reading_time: true
complexity: intermediate
description: "A linear solver computes the solution of a linear problem Ax = b. GEMSEO wraps the iterative solvers of SciPy and the MDAs use them to compute the coupled derivatives."
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

# Linear solvers { #concept-linear-solvers }

A linear solver computes the solution $x$ of a
[linear problem][concept-linear-problem] $Ax = b$.
GEMSEO wraps the iterative solvers of `scipy.sparse.linalg`.
The [catalog of linear solvers](https://gemseo.org/more_on/catalogs/linear_solvers/)
lists them with their settings.

!!! note
    Some solvers require a symmetric and positive definite left-hand side.
    The [LinearProblem][gemseo.linear.problem.LinearProblem] declares these properties.

??? note "Preconditioning"
    A preconditioner approximates the inverse of $A$.
    It is passed with the setting `preconditioner`.
    The setting `use_ilu_precond` builds one
    from an incomplete LU factorization of $A$,
    except when $A$ is a [LinearOperator][scipy.sparse.linalg.LinearOperator].
    The two settings cannot be used together.

??? note "Convergence failure"
    A solver that does not converge logs a warning; it does not raise an error.
    The setting `save_when_fail` saves the linear problem to the disk in that case.
    A solver that receives an illegal input or breaks down raises an error.

## Use in multidisciplinary analyses { #concept-linear-solvers-in-mda }

The MDAs solve linear systems involving $\partial R / \partial y$
to compute the [coupled derivatives][concept-adjoint-versus-direct-methods].
They solve one linear system per input component in direct mode
and one linear system per output component in adjoint mode.
These systems share the same left-hand side
and differ by their right-hand side.

The MDA setting `linear_solver_settings` defines the linear solver,
`LGMRES` by default.

With `linear_solver_settings=None`,
the MDA performs an LU factorization of $\partial R / \partial y$
and reuses it for all the right-hand sides.
If affordable,
this factorization may improve the solution time.
It is not possible when the MDA assembles $\partial R / \partial y$
as a linear operator.

!!! note
    The Newton-Raphson MDA also solves one linear system per iteration
    to compute the Newton step.
    The setting `newton_linear_solver_settings` defines its linear solver,
    `LGMRES` by default.

!!! tutorial
    - [Tutorial - Execute your first Multi-Disciplinary Optimization][]
