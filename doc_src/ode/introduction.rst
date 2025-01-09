..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
         :author: Isabelle Santos
         :author: Giulio Gargantini

:parenttoc: True
.. _introduction_to_ode:

Introduction to Ordinary Differential Equations
===============================================

Generalities
------------

ODE stands for ordinary differential equation.

The ODE module integrates the solution of first-order ordinary differential equations in |g|
by computing the solution of an Initial Value Problem (IVP).

An Initial Value Problem (IVP) is composed of the following entities:

* an initial state :math:`y_0 \in \mathbb{R}^n`;
* a time interval :math:`[t_0, t_f]`;
* a function :math:`f: [t_0, t_f] \times \mathbb{R}^n \rightarrow \mathbb{R}^n`.

The solution of an IVP consists in identifying the function :math:`y: [t_0, t_f]\rightarrow \mathbb{R}^n`
solving the following ODE:

.. math::

    \begin{cases}
    y(t_0) &= y_0 \\
    \frac{\mathrm{d} y}{\mathrm{d} t}(t) &= f(t, y(t)) \qquad \mbox{for all  }t \in [t_0, t_f].
    \end{cases}


Algorithms for the numerical solution
-------------------------------------

Multiple algorithms are available in the literature.
The `algorithms <../algorithms/ode_algos.html>`__ available in |g|, developed in the method ``solve_ivp``
of the library ``scipy.integrate``, are:

* the *explicit Runge-Kutta* algorithms (``RK45``, ``RK23``, ``DOP853``),
* an *implicit Runge-Kutta* algorithm (``Radau``),
* and two algorithms based on a backwards differentiation formula (``BDF`` and ``LSODA``).

The algorithms ``Radau``, ``BDF``, and ``LSODA`` require the knowledge of the Jacobian of
the function :math:`f` with respect to the state :math:`y`: :math:`J f = \frac{\partial f}{\partial y}`.
The Jacobian can be either passed to the algorithm, or computed by finite differences.

Further algorithms for the solution of IVPs are available in the plugin
`gemseo-petsc <https://gitlab.com/gemseo/dev/gemseo-petsc>`__.
The plugin `gemseo-petsc <https://gitlab.com/gemseo/dev/gemseo-petsc>`__ provides also an adjoint mode
to perform a sensitivity analysis on the solution of the ODE
with respect to its initial values and the design parameters.

.. figure:: /_images/algorithms/ODEProblem_ODEResult_attributes_description.png
Terminating events
------------------
For some problems, it might be interesting not to integrate the dynamic for the entire
time interval :math:`[t_0, t_f]`, but only up to the realization of a terminating condition.
Such conditions are encoded by **event functions**:
real-valued continuous functions :math:`g_1, \ldots, g_m: [t_0, t_f] \times \mathbb{R}^n \rightarrow \mathbb{R}`.
The terminating condition is realized when any of the event function crosses the threshold :math:`0`.
