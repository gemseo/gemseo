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
.. _ode_discipline:

The class ODEDiscipline
=======================

An :class:`.ODEDiscipline` is the subclass of :class:`.Discipline` wrapping an :class:`.ODEProblem`.

The function :math:`f(t, y)` defining the right-hand side of the ODE and the termination functions are encoded by
instances of :class:`.Discipline` with suitable inputs and outputs, allowing to couple different instances of
:class:`.ODEDiscipline` in an `MDA <../mdo/mda.html>`__.

Inputs and outputs
------------------

An instance of :class:`.ODEDiscipline` takes as inputs:

* the initial value of the *time* variable,
* the initial value of the *state* variables,
* the value of eventual *design variables*.

Without further specifications, the outputs of :class:`.ODEDiscipline` are the values of the state variables at the end
of the time interval (or, if *termination events* are present, at the realization of the first event).
By default, the name of the output variable corresponding to the final value of the state variable ``"y"`` is ``"y_final"``.

If the boolean field ``output_trajectory`` of the :class:`.ODEDiscipline` is set to ``True``, the discipline provides,
as additional outputs, the value of the state variables at the instants listed in the array ``times``.
By default, the output representing the trajectory of the state variable ``"y"`` is named ``"y"``.

Initialization
--------------
The instantiation of an :class:`.ODEDiscipline` requires at least two parameters: ``discipline``, representing the
function :math:`f(t, y)`, and ``times``, representing the time interval of integration of the ODE.
Further parameters can be specified at the time of the instantiation of the :class:`.ODEDiscipline`.

Coupling
--------
Like for other types of discipline, it is possible to couple instances of :class:`.ODEDiscipline` to other disciplines
in an `MDA <../mdo/mda.html>`__.
Coupled instances of :class:`.ODEDiscipline` can be used to model the dynamics of coupled physical
systems.

Coupled instances of ODEDiscipline
..................................
A first approach consists in modeling the different entities of the system by different instances of
:class:`.ODEDiscipline` with the parameter ``return_trajectories`` set to ``True``.
The coupling between the disciplines is done by passing the trajectories computed by each :class:`.ODEDiscipline` as
inputs of the other :class:`.ODEDiscipline` in the form of *design variables*.

.. image:: /_images/ode/coupling.png

Coupled dynamic inside an ODEDiscipline
.......................................
A different approach consists in defining a single :class:`.ODEDiscipline` for the entire system, having as state
variables the collection of all the variables representing each component of the coupled system, and as dynamic the
result of an *MDA* on all disciplines describing the dynamics of the components of the system.
variables the collection of all the variables representing each component of the coupled system.
The dynamic of such :class:`.ODEDiscipline` is the result of an `MDA <../mdo/mda.html>`__ on all disciplines describing
the dynamics of the components of the system.

.. image:: /_images/ode/time_integration.png


ODE Classes organization
--------------

Here is the UML diagram of the classes in |g| for the solution of ODEs.

.. uml::

    @startuml
    class ODEProblem
    class ODEDiscipline
    class ODEFunction
    class Discipline

    ODEDiscipline --|> Discipline
    ODEDiscipline "1" --* "1" ODEProblem
    ODEProblem "1" --* "n" ODEFunction

    ODEFunction "1" -- "1" Discipline
    @enduml

Examples
--------

See the examples about :class:`.ODEDiscipline` here:
`examples about ODEDiscipline <../examples/ode/index.html#odediscipline>`__.
