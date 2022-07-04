..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

.. _sellar_scalable:

Tutorial: Build a scalable model
====================================

In this tutorial,
we are going to build a scalable version
of the :class:`.Sellar1` :class:`.MDODiscipline`.
As a reminder, its expression reads:

:math:`y_1(x_{shared},x_{local},y_2)=\sqrt{x_{shared,1}^2+x_{shared,2}+x_{local}-0.2y_2`}.

.. seealso::

   - The Sellar's problem is described in :ref:`Sellar's problem <sellar_problem>`.
   - The scalable problem is described in :ref:`scalable`.

1. Overview
-----------

The expected outputs sizes are specified in a dictionary. The keys are the output names and the values are the sizes.
Here, we take 5 for the dimension of all outputs (here "y\_1", which is of dimension 1 in the standard :class:`.Sellar1`).

2. Creation of the discipline
-----------------------------

First of all, we create the reference :class:`.MDODiscipline`: with the help of the :class:`~gemseo.api.create_discipline` API function and the argument :code:`"Sellar1"`. As a reminder, this argument refers to the class :class:`.Sellar1`, which is internally known by |g| by means of the :class:`.DisciplinesFactory`.

.. code::

   from gemseo.api import create_discipline

   sellar = create_discipline("Sellar1")

.. tip::

   It is possible to implement a new discipline in a directory outside of the |g| directory (see :ref:`extending-gemseo`).
   Then, the user has to create a new class :class:`!NewMDODiscipline` inheriting from :class:`.MDODiscipline`.
   Then, the code reads:

   .. code::

      from gemseo.api import create_discipline

      newmdodiscipline = create_discipline("NewMDODiscipline")

Then, the scalable discipline can be created.

2. Creation of the scalable discipline
--------------------------------------

In |g|, scalable disciplines are defined by the class :class:`.ScalableDiscipline` that inherits from :class:`.MDODiscipline`.

Such a scalable discipline takes as mandatory arguments:

- a :code:`hdf_file_path` with its :code:`hdf_node_path` storing the evaluations of the :class:`.MDODiscipline`, here :code:`sellar`, over a :class:`.DiagonalDOE`,
- a :code:`sizes` dictionary describing the required sizes of inputs and outputs,
- a :code:`fill_factor` describing the probability of connection between an input and an output in the :class:`.ScalableDiscipline`,

and optional ones :

- a :code:`comp_dep` matrix (default: :code:`None`) that establishes the selection of a single original component for each scalable component,
- a :code:`inpt_dep` matrix (default: :code:`None`) that establishes the dependency of outputs w.r.t. inputs,
- a :code:`force_input_dependency` assertion (default: :code:`False`) describing that for any output, force dependency with at least on input,
- a :code:`allow_unused_inputs` assertion (default: :code:`False`) describing the possibility to have an input with no dependence with any output
- a :code:`seed` (default: :code:`1`)

2.1. Sample the discipline
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`hdf_file_path` file is built from the :meth:`~gemseo.api.create_scenario` API function applied to the :class:`.MDODiscipline` instance, :code:`sellar`,
with :code:`DOE` scenario type and the following :class:`.DesignSpace`:

.. code::

   from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace

   design_space = SellarDesignSpace()

The DOE algorithm is :code:`'DiagonalDOE'` and use a sampling of size :code:`n_samples=30`:

.. code::

   from gemseo.api import create_scenario

   sellar.set_cache_policy(cache_type='HDF5_cache', cache_tolerance=1e-6, cache_hdf_file='sellar.hdf5')
   output = sellar.get_output_data_names()[0]
   scenario = create_scenario([sellar], 'DisciplinaryOpt', output,
                              design_space, scenario_type='DOE')
   scenario.execute({'algo': 'DiagonalDOE', 'n_samples': 30})

A :class:`.DiagonalDOE` consists of equispaced points located on the diagonal of the design space.

2.2. Define the input and output dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A scalable discipline is a discipline version for which inputs and outputs can take arbitrary dimensions:

.. code::

   # Set the size of input and output variables at 5
   # - Number of n_x = number_of_inputs*variables_sizes
   # - Number of n_y = number_of_outputs*variables_sizes
   variables_sizes = 5
   input_names = sellar.get_input_data_names()
   output_names = sellar.get_output_data_names()
   sizes = {name: variables_sizes for name in input_names + output_names}

The :code:`sizes` of the inputs are specified in a dictionary at the construction of the :class:`.ScalableDiscipline` instance.

Lastly, we define the density factor for the matrix S describing the dependencies between the inputs and the outputs of the discipline:

.. code::

   # Density factor for the dependency matrix S
   fill_factor = 0.6

From this, we can create the :class:`.ScalableDiscipline` by means of the API function :meth:`~gemseo.api.create_discipline`:

.. code::

   # Creation of the scalable discipline
   scalable_sellar = create_discipline('ScalableDiscipline',
                                       hdf_file_path='sellar.hdf5',
                                       hdf_node_path='Sellar1',
                                       sizes=sizes,
                                       fill_factor=fill_factor)

3. Run the scalable discipline
------------------------------

After its creation,
the scalable discipline can be executed
by means of the :meth:`.MDODiscipline.execute` method.
For this,
we build an input dictionary.
Remember that the inputs and outputs shall all be in :math:`(0,1)` (see :ref:`scalable`).
Here we take :math:`( 0. ,  0.2,  0.4,  0.6,  0.8)`
for all inputs of the discipline ("x\_shared", "x\_local", and "y\_2").

.. code::

   from numpy import arange

   input_data = {name: arange(variables_sizes) / float(variables_sizes)
	             for name in input_names}
   print(scalable_sellar.execute(input_data)['y_1'])

The output of the discipline is:

.. code::

   [0.64353709  0.3085585   0.36497918  0.48043751  0.56740874]

of dimension 5, as expected.

Arbitrary input dimensions arrays can be provided. Here, only three components for all inputs and outputs are considered:

.. code::

    variables_sizes = 3
    sizes = {name: variables_sizes for name in input_names + output_names}
    scalable_sellar = create_discipline('ScalableDiscipline',
                                        hdf_file_path='sellar.hdf5',
                                        hdf_node_path='Sellar1',
                                        sizes=sizes,
                                        fill_factor=fill_factor)
    input_data = {name: arange(variables_sizes) / float(variables_sizes)
                  for name in input_names}

    print(scalable_sellar.execute(input_data)['y_1'])

The scalable discipline outputs different values :

.. code::

   [ 0.45727936  0.45727936  0.52084604]

We can see that multiple components of the output may be identical, because the original Sellar problem is of very low dimensions (1 or 2).
Therefore, the combinatorial effects that the scalable methodology uses to generate the outputs is not exploited (see :ref:`scalable`).
We obtain different output components in higher dimension.

4. Perspectives
---------------

This :class:`.ScalableDiscipline` can now be included as any other in a :class:`.MDOScenario` to compare the scalability of MDO or coupling strategies.

Such a :class:`.ScalableDiscipline` as two main advantages:

- The execution time shall be very small even for thousands of inputs and outputs.
- Analytical derivatives are also available (Jacobian matrices), even if the original discipline has no analytic derivatives.
