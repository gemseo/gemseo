..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Isabelle Santos

Ordinary Differential Equations
-------------------------------

An :class:`.ODEProblem` represents an ordinary differential equation (ODE) with a given
state at an initial time.
This :class:`.ODEProblem` is built with a function of time and state, as well as an array
describing the intial state.

.. note::

    This feature is under active development. Future iterations include the integration of
    :class:`.ODEProblem` s with :class:`.MDODiscipline`.
