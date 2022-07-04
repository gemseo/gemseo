..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

.. _dev_scalable:

Scalable models
===============

.. automodule:: gemseo.problems.scalable.data_driven.api
   :noindex:

.. automodule:: gemseo.problems.scalable.data_driven.problem
   :noindex:

.. automodule:: gemseo.problems.scalable.data_driven.discipline
   :noindex:

.. automodule:: gemseo.problems.scalable.data_driven.factory
   :noindex:

.. automodule:: gemseo.problems.scalable.data_driven.model
   :noindex:

.. automodule:: gemseo.problems.scalable.data_driven.diagonal
   :noindex:

.. uml::

   MDODiscipline <|-- ScalableDiscipline
   ScalableDiscipline *-- ScalableModel
   ScalableModel <|-- ScalableDiagonalModel
   ScalableDiagonalModel *-- ScalableApproximation

   class ScalableDiscipline {
    +scalable_model
    +initialize_grammars()
	#compute_jacobian()
	#run()
	#set_default_inputs()
   }

   class ScalableModel {
    +cache
    +lower_bounds
    +upper_bounds
    +model
    +name
    +parameters
    +sizes
    +build_model()
    +compute_bounds()
    +inputs_names()
    +outputs_names()
    +original_sizes()
    +scalable_function()
    +scalable_derivatives()
    #set_sizes()
   }
