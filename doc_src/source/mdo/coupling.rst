..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Matthias De Lozzo

Coupling visualization
======================

First of all, we need to instantiate the different :class:`~gemseo.core.discipline.MDODiscipline`, e.g.

.. code::

    from gemseo.api import create_discipline

    disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])

N2 chart
--------

Then, we can represent the coupling structure of these :code:`disciplines` by means of a **N2 chart**,
also referred to as N2 diagram, N-squared diagram or N-squared chart.

.. seealso::

   More information concerning the N2 chart `on wikipedia <https://en.wikipedia.org/wiki/N2_chart>`_.

Description
~~~~~~~~~~~

For that, we consider the :meth:`~gemseo.api.generate_n2_plot` API function:

.. code::

    from gemseo.api import generate_n2_plot

    generate_n2_plot(disciplines)

and obtains the following N2 chart where:

- disciplines are on the diagonal,
- the input coupling variables of a discipline are located at the vertical of the discipline,
- the output coupling variables of a discipline are located at the horizontal of the discipline.

In this case, Sellar1 returns the coupling variable y_0 which is an input for Sellar2 and SellarSystem.
Similarly, Sellar2 returns the coupling variable y_1 which is an input for Sellar1 and SellarSystem.


.. figure:: /_images/coupling/n2.png
   :scale: 75 %

Options
~~~~~~~

The first argument of the :meth:`~gemseo.api.generate_n2_plot` API function is the :code:`list` of :class:`~gemseo.core.discipline.MDODiscipline`.
This argument is mandatory while the others are optional:

- :code:`file_path`: file path of the figure (default value: :code:`'n2.pdf'`)
- :code:`show_data_names`: if true, the names of the coupling data is shown otherwise, circles are drawn, which sizes depend on the number of coupling names (default value: :code:`True`)
- :code:`save`: if True, saved the figure to file_path (default value: :code:`True`)
- :code:`show`: if True, shows the plot (default value: :code:`False`)
- :code:`figsize`: tuple, size of the figure (default value: :code:`(15,10)`)

Here, when :code:`show_data_names` is :code:`False`, we obtain:

.. figure:: /_images/coupling/n2_disc.png
   :scale: 75 %

Coupling graph
--------------

We can also represent this relation by means of a **coupling graph**.

Description
~~~~~~~~~~~

For that, we consider the :meth:`~gemseo.api.generate_coupling_graph` API function:

.. code::

    from gemseo.api import generate_coupling_graph

    generate_coupling_graph(disciplines)

and obtains the following coupling graph where:

- disciplines are represented by bubbles,
- coupling variable flows are represented by arrows between disciplines,
- constraint and objectives functions are represented.

In this case, the coupling variable y_0 goes from Sellar1 to Sellar2 and from Sellar1 to SellarSystem.
Similarly, the coupling variable y_1 goes from Sellar2 to Sellar1 and from Sellar2 to SellarSystem.
Moreover, SellarSystem returns the value of the objective function and constraints.

.. figure:: /_images/coupling/coupling_graph.png

Options
~~~~~~~

The first argument of the :meth:`~gemseo.api.generate_coupling_graph` API function is the :code:`list` of :class:`~gemseo.core.discipline.MDODiscipline`.
This argument is mandatory while the other is optional:

- :code:`file_path`: file path of the figure (default value: :code:`'coupling_graph.pdf'`)
