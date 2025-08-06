..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

Aerostructure problem
---------------------

The Aerostructure problem is considered in the :ref:`sphx_glr_examples_scalable_plot_problem.py` example.

Description of the problem
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Aerostructure problem is defined by analytical functions:

.. include:: aerostructure_problem_definition.inc

The Aerostructure disciplines are also available with analytic derivatives in the classes :class:`.Mission`,
:class:`.Aerodynamics` and :class:`.Structure`, as well as a :class:`.AerostructureDesignSpace`:

Creation of the disciplines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the aerostructure disciplines, use the function :func:`.create_discipline`:

.. code::

     from gemseo import create_discipline

     disciplines = create_discipline(["Aerodynamics", "Structure", "Mission"])

Importation of the design space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.AerostructureDesignSpace` class can be imported as follows:

.. code-block:: python

    from gemseo.problems.aerostructure.aerostructure_design_space import AerostructureDesignSpace
    design_space = AerostructureDesignSpace()

Then, you can visualize it with ``print(design_space)``:

.. code-block:: bash

    +----------------+-------------+-------------+-------------+-------+
    | name           | lower_bound |    value    | upper_bound | type  |
    +----------------+-------------+-------------+-------------+-------+
    | thick_airfoils |      5      |   (15+0j)   |      25     | float |
    | thick_panels   |      1      |    (3+0j)   |      20     | float |
    | sweep          |      10     |   (25+0j)   |      35     | float |
    | drag           |     100     |   (340+0j)  |     1000    | float |
    | forces         |    -1000    |   (400+0j)  |     1000    | float |
    | lift           |     0.1     |   (0.5+0j)  |      1      | float |
    | mass           |    100000   | (100000+0j) |    500000   | float |
    | displ          |    -1000    |  (-700+0j)  |     1000    | float |
    | rf             |    -1000    |      0j     |     1000    | float |
    +----------------+-------------+-------------+-------------+-------+

.. seealso::

   See :ref:`sphx_glr_examples_scalable_plot_problem.py` to see an application of this problem.
