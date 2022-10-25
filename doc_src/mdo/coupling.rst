..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Matthias De Lozzo

.. _coupling_visualization:

Coupling visualization
======================

|g| offers two ways of visualizing a multidisciplinary coupling structure,
either using a network diagram based on nodes and links
or using an N2 chart based on a tabular view.

Dependency graph
----------------

Both rely on the same :class:`.DependencyGraph`,
built from the library `NetworkX <https://networkx.org/>`_.
This :class:`.DependencyGraph`
generates not only the full
`directed graph <https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)>`_
but also the condensed one
with the Tarjan's algorithm [1]_ with Nuutila's modifications [2]_.

.. [1] Depth-first search and linear graph algorithms, R. Tarjan
   SIAM Journal of Computing 1(2):146-160, (1972).

.. [2] On finding the strongly connected components in a directed graph.
   E. Nuutila and E. Soisalon-Soinen
   Information Processing Letters 49(1): 9-14, (1994).

The following example is used to illustrate these features:

.. code::

    from gemseo.api import create_discipline

    disc0 = create_discipline('AnalyticDiscipline', name='D0', expressions={'y0':'x0+y10+y2'})
    disc1 = create_discipline('AnalyticDiscipline', name='D1', expressions={'y10':'x0+x1+y2+y10', 'y11':'x0-x1+2*y11'})
    disc2 = create_discipline('AnalyticDiscipline', name='D2', expressions={'y2':'x0+x2+y10'})
    disciplines = [disc0, disc1, disc2]

where the disciplines D1 and D2 are strongly coupled
while D0 is weakly coupled to D1 and D2.
Moreover,
D1 is self-coupled,
i.e. one of its outputs is also one of its inputs.

Coupling graph visualization
----------------------------

Both full and condensed graphs can be represented as network diagrams
where disciplines are nodes represented by circles labeled by their names,
and couplings are links represented by arrows labeled by their coupling variables.

API
~~~

The API function :meth:`~gemseo.api.generate_coupling_graph` allows
to create these visualizations and save them, from:

- the :class:`.MDODiscipline` instances defining the disciplines of interest,
- a file path
  (by default,
  the default current working directory with *coupling_graph.pdf* as file name),
- and the type of graph to display
  (by default,
  the full graph).

Full graph
~~~~~~~~~~

.. code::

    from gemseo.api import generate_coupling_graph

    generate_coupling_graph(disciplines)

.. figure:: /_images/coupling/full_coupling_graph.png


Condensed graph
~~~~~~~~~~~~~~~

.. code::

    generate_coupling_graph(disciplines, full=False)

.. figure:: /_images/coupling/condensed_coupling_graph.png

N2 chart visualization
----------------------

Both full and condensed graphs can be represented
as `N2 charts <https://en.wikipedia.org/wiki/N2_chart>`_
also referred to as N2 diagrams, N-squared diagrams or N-squared charts.

The diagonal elements of an N2 chart are the disciplines
while the non-diagonal elements are the coupling variables.
A discipline takes its inputs vertically and returns its outputs horizontally.
In other words,
if the cell *(i,j)* is not empty,
its content is the set of the names of the variables
computed by the *i*-th discipline and passed to the *j*-th discipline.

|g| offers the possibility to display the N2 chart
either as a static visualization of the full graph,
or as an interactive visualization of both full and condensed graphs:

.. note::

    The self-coupled disciplines are represented by diagonal blocks
    with a specific background color:
    blue for the static N2 chart and
    the color of the group to which the discipline belongs for the dynamic N2 chart.

API
~~~

The API function :meth:`~gemseo.api.generate_n2_plot` allows
to create these visualizations and save them, from:

- the :class:`.MDODiscipline` instances defining the disciplines of interest,
- a file path
  (by default,
  the default current working directory with *n2.pdf* as file name),
- whether to display the names of the coupling variables in the static N2 chart
  (by default, True),
- whether to save the static N2 chart
  (by default, True),
- whether to show the static N2 chart in a dedicated window
  (by default, True),
- the size of the figure of the static N2 chart
  (by default, width equal to 15 and height equal to 10),
- and whether to open the default web browser and display the interactive N2 chart
  (by default, False).

Whatever the options,
an HTML file is create based on the provided file path
by using *.html* as file extension (by default, *n2.html*):

.. figure:: /_images/coupling/n2.gif
   :scale: 75 %
   :target: ../_static/n2.html

This interactive N2 chart can be opened at any time in a browser

.. seealso::

   `Click here <../_static/n2.html>`_ to see the example of an interactive N2 chart
   with several groups of strongly coupled disciplines

With coupling names
~~~~~~~~~~~~~~~~~~~

.. code::

    from gemseo.api import generate_n2_plot

    generate_n2_plot(disciplines)

.. figure:: /_images/coupling/n2.png
   :scale: 75 %

Without coupling names
~~~~~~~~~~~~~~~~~~~~~~

.. code::

    from gemseo.api import generate_n2_plot

    generate_n2_plot(disciplines, show_data_names=False)

.. figure:: /_images/coupling/n2_without_names.png
   :scale: 75 %
