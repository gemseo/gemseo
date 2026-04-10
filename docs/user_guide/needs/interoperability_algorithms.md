<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Interoperability of algorithms { #needs-interoperability-of-algorithms }

## The problem { #needs-the-problem }

Solving a multidisciplinary design problem
requires a variety of numerical algorithms:
optimization solvers, DOE methods, MDA solvers, linear solvers, surrogate models, etc.
Each type of algorithm comes from different libraries
(SciPy, NLopt, OpenTURNS, scikit-learn, pyOptSparse, ...),
each with its own API, conventions, and configuration style.

Switching from one optimizer to another,
or combining a DOE with a surrogate model and then an optimizer,
should not require rewriting the problem formulation.
Yet in practice,
every library has a different interface,
making it hard to experiment with alternatives
or to combine methods.

## GEMSEO's answer: a unified algorithm interface { #needs-gemseos-answer-a-unified-algorithm-interface }

GEMSEO wraps all algorithms behind a **uniform interface**
based on the factory pattern.
Each category of algorithm is managed by a dedicated library class:

- **Optimization**: 30+ algorithms from SciPy, NLopt, pyOptSparse, and GEMSEO's own implementations.
- **Design of experiments**: Monte Carlo, LHS, full factorial, low-discrepancy sequences, etc. from PyDOE and OpenTURNS.
- **MDA solvers**: Gauss-Seidel, Jacobi, Newton-Raphson, quasi-Newton, and hybrid methods.
- **Linear solvers**: direct and iterative methods.
- **ODE solvers**: for dynamic systems.
- **Machine learning**: regression, classification, clustering from scikit-learn and OpenTURNS.

<!-- ![Optimization concept](../../assets/images/user_guide/optimization_concept.png){: style="display:block; margin:auto; max-width:70%" } -->

*GEMSEO formulates the problem; disciplines evaluate it; wrapped libraries solve it.*

All algorithms are accessed through the same pattern:
the user specifies an algorithm **by name** and provides its **settings**
via a Pydantic model.
Switching from one algorithm to another
is a matter of changing a single name and its settings,
without modifying any other part of the process.

This also means algorithms can be **combined freely**:
an optimizer can drive a process that uses an MDA solver internally,
which itself can be accelerated by a surrogate model.
GEMSEO handles the plumbing.

The algorithm catalog is **extensible**:
new algorithms can be added via plugins
without modifying GEMSEO's core code.
GEMSEO discovers them automatically at runtime through its factory mechanism.
