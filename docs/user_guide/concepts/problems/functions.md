---
description: "Array functions map input arrays to output arrays and support arithmetic operators with automatic differentiation."
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

# Functions { #concept-functions }

An array function is a mathematical function that maps an input array to an output array,
i.e. any function of the form
$f: \mathbb{C}^n \rightarrow \mathbb{C}^m, x \mapsto y = f(x)$.

In GEMSEO, this concept is implemented by the
[ArrayFunction][gemseo.core.functions.array_function.ArrayFunction] object.
The mapping from $x$ to $y$ is a user-defined callable (e.g. a Python function)
provided at instantiation via the `func` argument
and stored in the `func` attribute.
However,
it is recommended to evaluate the array function using the
[evaluate()][gemseo.core.functions.array_function.ArrayFunction.evaluate] method,
which can perform post-processing,
such as calculating execution statistics
(if the function is a [ProblemFunction][gemseo.algos.problem_function.ProblemFunction])
or extracting the real part when dealing with an imaginary number.

The [ArrayFunction][gemseo.core.functions.array_function.ArrayFunction] can also
be given a callable to compute its Jacobian via the `jac` argument,
which is then stored in `jac` attribute.

Array functions support all the essential arithmetic operations
such as addition, subtraction, multiplication, division and negation.
These operations support automatic differentiation,
provided both operands have a Jacobian callable in `jac`.

Examples of specific array functions typically include:

- linear and quadratic functions from coefficients vectors/matrices,
- first and second-order Taylor polynomials from an array function,
- restriction from an array function and input components.
