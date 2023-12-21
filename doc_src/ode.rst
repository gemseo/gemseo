..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Isabelle Santos

Ordinary Differential Equations (ODE)
-------------------------------------

ODE stands for Ordinary Differential Equation.


.. seealso::

   `The examples related to ODE resolution. <../examples/ode/index.html>`__


An :class:`.ODEProblem` represents a first order ordinary differential equation (ODE) with
a given state at an initial time.
This :class:`.ODEProblem` is built with a function of time and state, as well as an array
describing the intial state, and a time interval.

An :class:`ODEResult` represents the solution of an ODE evaluated at a discrete set of
times within the specified time interval.


.. note::

    This feature is under active development. Future iterations include the integration of
    :class:`.ODEProblem` s with :class:`.MDODiscipline`.


Architecture
~~~~~~~~~~~~


ODEProblem and ODEResult
........................

The main classes in the ODE submodule are the :class:`.ODEProblem` and :class:`ODEResult`.
These represent respectively the first-order ODE with its initial conditions, and the
solution of this problem evaluated at a discrete set of values for time.

As a reminder, a first-order ordinary differential equation is an equation of the form:

.. math::

    \frac{ds}{dt}(t) = f(t, s(t)) \ \textrm{ and }\ s(t_0) = s_0

where :math:`s` is the state which depends on :math:`t`, the time. The right-hand side
function :math:`f` is a function of the time and the state. The value of the state at an
initial time :math:`t_0` is known to be :math:`s_0`.

The solution of this problem is provided for discrete values of time within a given
interval :math:`[t_0,\ t_f]`.

.. uml::

    @startuml
    set namespaceSeparator none
    class "ODEProblem" as gemseo.algos.ode.ode_problem.ODEProblem {
    rhs_function
    initial_state
    intitial_time
    final_time
    algorithm_name
    }
    class "ODEResult" as gemseo.algos.ode.ode_result.ODEResult {
    state_vector
    time_vector
    is_converged
    }

    gemseo.algos.ode.ode_problem.ODEProblem *-- gemseo.algos.ode.ode_result.ODEResult : result
    @enduml


.. figure:: /_images/algorithms/ODEProblem_ODEResult_attributes_description.png

    Correspondance between the elements of an ordinary differential equation with initial
    conditions and the attributes of the :class:`ODEProblem` and :class:`ODEResult` classes.


Classes
.......

The classes described by the ODE module are as such:

.. uml::

    @startuml
    set namespaceSeparator none
    class "ODEProblem" as gemseo.algos.ode.ode_problem.ODEProblem {}
    class "ODEResult" as gemseo.algos.ode.ode_result.ODEResult {}
    class "ODESolverLib" as gemseo.algos.ode.ode_solver_lib.ODESolverLib {
    }
    class "ODESolversFactory" as gemseo.algos.ode.ode_solvers_factory.ODESolversFactory {
      execute(problem: ODEProblem, algo_name: str) -> ODEResult
    }
    class "ScipyODEAlgos" as gemseo.algos.ode.lib_scipy_ode.ScipyODEAlgos {

    }
    gemseo.algos.ode.lib_scipy_ode.ScipyODEAlgos --|> gemseo.algos.ode.ode_solver_lib.ODESolverLib
    gemseo.algos.ode.ode_result.ODEResult --* gemseo.algos.ode.ode_problem.ODEProblem : result
    gemseo.algos.ode.ode_solver_lib.ODESolverLib --* gemseo.algos.ode.ode_solvers_factory.ODESolversFactory
    @enduml


Packages
........

The submodules are organized in the following fashion.

.. uml::

    @startuml packages
    set namespaceSeparator none
    package "gemseo.algos.ode" as gemseo.algos.ode {
    }
    package "gemseo.algos.ode.lib_scipy_ode" as gemseo.algos.ode.lib_scipy_ode {
    }
    package "gemseo.algos.ode.ode_problem" as gemseo.algos.ode.ode_problem {
    }
    package "gemseo.algos.ode.ode_result" as gemseo.algos.ode.ode_result {
    }
    package "gemseo.algos.ode.ode_solver_lib" as gemseo.algos.ode.ode_solver_lib {
    }
    package "gemseo.algos.ode.ode_solvers_factory" as gemseo.algos.ode.ode_solvers_factory {
    }
    gemseo.algos.ode.lib_scipy_ode --> gemseo.algos.ode.ode_result
    gemseo.algos.ode.lib_scipy_ode --> gemseo.algos.ode.ode_solver_lib
    gemseo.algos.ode.ode_problem --> gemseo.algos.ode.ode_result
    gemseo.algos.ode.ode_solver_lib --> gemseo.algos.ode.ode_problem
    gemseo.algos.ode.ode_solvers_factory --> gemseo.algos.ode.ode_problem
    gemseo.algos.ode.ode_solvers_factory --> gemseo.algos.ode.ode_solver_lib
    @enduml
