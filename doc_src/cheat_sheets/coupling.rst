..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

==================
Coupling structure
==================

.. code-block:: python

    from gemseo import create_discipline
    from gemseo import generate_coupling_graph
    from gemseo import generate_n2_plot
    from gemseo.disciplines.utils import get_all_inputs
    from gemseo.disciplines.utils import get_all_outputs

Save or show the N2 chart:

.. code-block:: python

    discipline_names = ("disc1", "disc2", "disc3")
    disciplines = create_discipline(discipline_names)
    generate_n2_plot(disciplines, save=True, show=False)

Save the coupling graph:

.. code-block:: python

    discipline_names = ("disc1", "disc2", "disc3")
    disciplines = create_discipline(discipline_names)
    generate_coupling_graph(disciplines)

Get all the inputs or outputs:

.. code-block:: python

    get_all_inputs(disciplines)
    get_all_outputs(disciplines, skip_scenarios=False)
