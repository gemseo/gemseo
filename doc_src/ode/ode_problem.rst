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
.. _ode_problem:

The class ODEProblem
====================

Presentation
------------
The class :class:`.ODEProblem` is used to represent an initial value problem (IVP) in the form:

.. math::

    \begin{cases}
    &\text{Find the function } y(t) \\
    &\text{defined on the time interval }[t_0, t_f] \\
    &\text{such that }y(0) = y_0, \\
    &\text{and }\frac{\mathrm{d}}{\mathrm{d}t}y(t) = f(t, y(t))\text{  for all } t \in [t_0, t_f].
    \end{cases}

The function :math:`f: t, y \mapsto f(t, y)` defines the dynamics of the problem, by computing the
time derivative of the state variable :math:`y` at time :math:`t`.
The term :math:`y_0` identifies the state of the time variable at the initial time :math:`t_0`.

If no termination function is defined, the function :math:`y(\cdot)`
is computed on the time interval :math:`[t_0, t_f]`.
Otherwise, it is possible to define a list of **termination functions** :math:`g_1, \ldots, g_m`
taking as arguments the time :math:`t` and the state :math:`y`.
In such a case, the function :math:`y(\cdot)` is the solution of the following problem:

.. math::

    \begin{cases}
    &\text{Find the function } y(t) \\
    &\text{defined on the time interval }[t_0, t^*] \\
    &\text{such that }y(0) = y_0,\\
    &\frac{\mathrm{d}}{\mathrm{d}t}y(t) = f(t, y(t))\text{  for all } t \in [t_0, t_f], \\
    &\text{and }t^* \text{is the smallest element of }[t_0, t_f] \\
    &\text{for which } g_i(t^*, y(t^*)) = 0 \text{ for some } i \in \lbrace 1, \ldots, m\rbrace.
    \end{cases}

Initialization
--------------
In order to instantiate an :class:`.ODEProblem`, the following arguments are required:

*   ``func``: a python function or a functor taking as arguments the time variable and the state variable,
    identifying the dynamic of the problem;
*   ``initial_state``: an *ArrayLike* variable of the same dimension as the state variable,
    identifying the initial condition of the IVP;
*   ``times``: an *ArrayLike* of ``float``, whose extremities identify the time interval :math:`[t_0, t_f]`.

It should be remarked that the dynamic of the IVP may depend on a set of parameters named
**design variables**, which remain constant during the solution of the IVP.
Different values of the design variables correspond to different functions :math:`f(t, y)`, and yield
different solutions of the IVP.

Further optional arguments can be added at the time of the instantiation of an :class:`.ODEProblem`
in order to enrich the IVP with more complex terminating conditions, or to ease the solution of the IVP.


Solution of the IVP
-------------------
An instance of :class:`.ODEProblem` is used to represent an IVP.
In order to solve it, it is necessary to instantiate an :class:`.ODESolverLibraryFactory` and execute it:

.. code::
    ODESolverLibraryFactory().execute(problem=problem, algo_name=algo_name, **kwargs)

The method ``execute`` of :class:`ODESolverLibraryFactory` takes as arguments the :class:`.ODEProblem`
to be solved, the algorithm to use, and eventual other keyword parameters that are necessary for the
execution of the chosen algorithm.

The method ``execute`` of :class:`ODESolverLibraryFactory` computes the solution of the IVP and stores
it in ``problem.result`` as an instance of the data class :class:`ODEResult`.

.. code::

    ODESolverLibraryFactory().execute(ode_problem, algo_name="RK45")


Examples
--------

See the examples about the class :class:`.ODEProblem` here:
`ODEProblem examples <../examples/ode/index.html#odeproblem-and-oderesult>`__.
