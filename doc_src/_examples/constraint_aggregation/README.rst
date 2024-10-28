..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.


Constraint aggregation
======================
In this section the use of constraint aggregation is demonstrated.
We will focus on the following problem:

.. math:: \min_{x \in [0,1]}{(\max_{j=1,...,N}{(f_j(x)})}

where:

.. math:: f_j(x) = jxe^{1-jx} \quad \forall j=1,...,N


This can be reformulated into a non-linear optimization problem with non linear constraints:

.. math:: \min{(y)}
.. math:: s.t.
.. math:: g_j(x,y) = f_j(x)-y \leq 0 \quad \forall j=1,...,N
.. math:: 0\leq x \leq 1


In this situation this problem can be solved using any non linear optimization solver supporting non linear inequality constraints.
The solution of this problem is :math:`(x^*,y^*) \equiv (0,0)\quad \forall N\in \mathbf{N}` .

Some optimization solvers, such as MMA, can become expensive when the number of constraints :math:`N` is large.
Moreover, since constraint gradients are often needed, reducing the number of constraints to fewer constraints can be efficient to compute the gradient by adjoint approach.
Constraint aggregation aims to reduce the number of constraints in an optimization problem. It does this by replacing the original constraints with operators that transform vectors into scalars.

.. math:: \min{(y)}
.. math:: s.t.
.. math:: G_{agg}(g_j(x,y), p) \leq 0
.. math:: 0\leq x \leq 1

Where :math:`G_{agg}(g_j(x,y), p)` indicates an aggregation operator and  :math:`p` its parameters (if applicable).
In the next section some aggregation approaches are reviewed.

The maximum
-----------
The simplest approach to aggregate the constraint functions is to take their maximum.

.. math:: G_{max}(g_j(x,y), p) = \max_{j=1,...,N}{(f_j(x)})

using this formulation keeps the same solution of the original problem but has the major inconvinient of not being differentiable.
This is not always a problem in practice. However, it is possible that the performance of gradient-based solvers may be degraded.

The KS function
---------------
The Kreisselmeier–Steinhauser (KS) function :cite:`kreisselmeier1983application` is a continuous and differentiable function that tends to the maximum operator when the aggregation parameter tends to infinity.


.. math:: G_{KS}(g_j(x,y), p) = \frac{1}{p}\log({\sum_{j=1}^{N}{\exp(pg_j(x,y))}})

It can be shown that:

.. math:: G_{KS}(g_j(x,y), p)-G_{max}(g_j(x,y), p)\leq \frac{\log(N)}{p}

This means that using KS function aggregation the solution of the optimization problem is an approximation of the original problem.
Since the KS function is always greater than the maximum, this approach always leads to feasible solutions.
By increasing the value of p, the KS function will be closer to the maximum and the optimization will increase the number of iterations to converge.

The Induced Exponential
-----------------------
The Induced Exponential function :cite:`kennedy2015improved` is also a continuous and differentiable function that tends to the maximum operator when the aggregation parameter tends to infinity.

.. math:: G_{IKS}(g_j(x,y), p) = \frac{\sum_{j=1}^{N}{g_j(x,y)e^{pg_j(x,y)}}}{\sum_{j=1}^{N}{e^{pg_j(x,y)}}}

It can be shown that:

.. math:: G_{max}(g_j(x,y), p)-G_{IKS}(g_j(x,y), p)\leq \frac{\log(N)}{p}

This means that using IKS function aggregation, the solution to the optimization problem is an approximation of the original problem.
Since the IKS function is always smaller than the maximum, this approach always leads to infeasible solutions.
By increasing the value of p, the IKS function will be closer to the maximum and the optimization will increase the number of iterations to converge.
