..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

For a design problem to be solved, and given separated :term:`simulation software` (called :term:`disciplines <discipline>`),
this generic engine for :ref:`MDO scenarios <mdo_formulations>`:

	#. creates a :term:`scenario` that translates the design problem into a mathematical :term:`optimization problem`,

		- The user can choose its favorite :ref:`MDO formulation <mdo_formulations>` (or architecture) or test different ones,

	#. connects this optimization problem to the given disciplines and to an :term:`optimization algorithm`,

		- Within a scenario, :ref:`Bi-level MDO formulations <bilevel_formulation>` allow to use disciplines that are themselves scenarios.

	#. solves this optimization problem and returns the design problem solution,

		- The resolution of the optimization problem can also be driven by a :term:`DOE` algorithm, rather than an optimization one.

	#. post-processes the results and logs of the optimization problem resolutions.
