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

The Sobieski's SSBJ test case is considered in the different tutorials:

- :ref:`aerostruct_toy_example`

Description of the problem
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Aerostructure problem is defined by analytical functions:

.. include:: aerostructure_problem_definition.inc

The Aerostructure disciplines are also available with analytic derivatives in the classes :class:`~gemseo.problems.aerostructure.aerostructure.Mission`, :class:`~gemseo.problems.aerostructure.aerostructure.Aerodynamics` and :class:`~gemseo.problems.aerostructure.aerostructure.Structure`, as well as a :class:`~gemseo.problems.aerostructure.aerostructure_design_space.AerostructureDesignSpace`:

Creation of the disciplines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the aerostructure disciplines, use the function :func:`.create_discipline`:

.. code::

     from gemseo import create_discipline

     disciplines = create_discipline(["Aerodynamics", "Structure", "Mission"])

Importation of the design space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To import the aerostructure design space, use the class ``create_discipline``:

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

   See :ref:`aerostruct_toy_example` to see an application of this problem.
