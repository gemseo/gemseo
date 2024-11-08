..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.


Exterior Penalty
================

The exterior penalty is used to reformulate an optimization problem with both equality and inequality nonlinear constraints into one without constraints.
We focus on optimization problems in the following general form:

.. math:: \min_x{f(x)}
.. math:: s.t.
.. math:: g_i(x)\leq0 \quad \forall i=1,2,...,m
.. math:: h_j(x)=0 \quad \forall j=1,2,...,p
.. math:: l_b \leq x\leq u_b

Where :math:`x` is the design variable vector, :math:`f` is the objective function, :math:`g` the inequality constraint functions, :math:`h` the equality constraint functions, :math:`l_b` the lower bounds and :math:`u_b` the upper bounds.
This problem can be approximated by:

.. math:: \min_x{\left(\frac{f(x)}{f_0}+\rho_{ineq}\sum_{i=1}^{m}{H(g_i(x))g_i(x)^2}+\rho_{eq}\sum_{j=1}^{p}{h_i(x)^2}\right)}
.. math:: s.t.
.. math:: l_b \leq x\leq u_b

Where :math:`\rho_{ineq}` is the penalty constant for inequality constraints, :math:`\rho_{eq}` is the penalty constant for equality constraints, :math:`f_0` is the objective scaling factor and :math:`H` indicates the heaviside function.
The reformulated problem solution can only be considered as an approximation of the original problem.
The solution to this problem will always violate the constraints. This fact gives this approach its name, since the solution of the reformulated problem approximates the solution from outside the feasible design domain.
Increasing the value of :math:`\rho_{ineq}`, :math:`\rho_{eq}` and of :math:`f_0` the solution of this problem will tend to the original one.
The reformulated problems become ill-conditioned if these values are increased too much. This can affect the performance of the optimization solver.
