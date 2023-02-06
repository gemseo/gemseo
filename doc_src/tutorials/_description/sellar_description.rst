..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

Sellar's problem
----------------

The Sellar's problem is considered in different tutorials:

- :ref:`sphx_glr_examples_mdo_plot_gemseo_in_10_minutes.py`
- :ref:`sellar_mdo`

Description of the problem
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Sellar problem is defined by analytical functions:

.. include:: /tutorials/_description/sellar_problem_definition.inc

The Sellar disciplines are also available with analytic derivatives in |g|, as well as a :class:`~gemseo.algos.design_space.DesignSpace`:

Creation of the disciplines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the Sellar disciplines, use the function ``create_discipline``:

.. code::

     from gemseo.api import create_discipline

     disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])

Importation of the design space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To import the Sellar design space, use the class ``create_discipline``:

.. code-block:: python

    from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
    design_space = SellarDesignSpace()

Then, you can visualize it with :code:`print(design_space)`:

.. code-block:: bash

    +----------+-------------+--------+-------------+-------+
    | name     | lower_bound | value  | upper_bound | type  |
    +----------+-------------+--------+-------------+-------+
    | x_local  |      0      | (1+0j) |      10     | float |
    +          +             +        +             +       +
    | x_shared |     -10     | (4+0j) |      10     | float |
    +          +             +        +             +       +
    | x_shared |      0      | (3+0j) |      10     | float |
    +          +             +        +             +       +
    | y_1      |     -100    | (1+0j) |     100     | float |
    +          +             +        +             +       +
    | y_2      |     -100    | (1+0j) |     100     | float |
    +----------+-------------+--------+-------------+-------+

.. seealso::

   See :ref:`sellar_mdo` to create the Sellar problem from scratch
