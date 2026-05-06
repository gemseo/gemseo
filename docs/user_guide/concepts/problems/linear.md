---
description: "A linear problem represents a system of linear equations Ax = b to be solved for x given a matrix or operator A and a right-hand side vector b."
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

# Linear problem { #concept-linear-problem }

A [LinearProblem][gemseo.algos.linear_solvers.linear_problem.LinearProblem]
represents a system of linear equations of the form $Ax = b$,
where $A$ is a matrix or a linear operator referred to as the *left-hand side* (LHS),
$b$ is a vector referred to as the *right-hand side*,
and $x$ is the solution vector to be determined.

The LHS can have several formats:

- dense matrices (NumPy arrays),
- sparse matrices (SciPy sparse arrays),
- [LinearOperator][scipy.sparse.linalg.LinearOperator] (matrix-free setting).

Additional information on the LHS structure can be provided at instantiation, such as
symmetry or positive definiteness via the `is_symmetric` and `is_positive_def`
arguments respectively.

Once solved,
the `solution` attribute holds $x$
and `is_converged` indicates whether the solver reached the specified tolerance.
The relative residual $\|Ax - b\|_2 / \|b\|_2$
or the absolute residual $\|Ax - b\|_2$
can be computed via
[compute_residuals()][gemseo.algos.linear_solvers.linear_problem.LinearProblem.compute_residuals].
Passing `store=True`
to [compute_residuals()][gemseo.algos.linear_solvers.linear_problem.LinearProblem.compute_residuals]
appends each value to residuals_history,
which can then be plotted
with [plot_residuals()][gemseo.algos.linear_solvers.linear_problem.LinearProblem.plot_residuals]
to inspect the convergence of iterative solvers.

!!! note

    The linear problem can be solved automatically by a linear solver.
    Please refer to the [linear solvers section][concept-linear-solvers] for more information.
