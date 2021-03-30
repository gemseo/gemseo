..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _radar_chart:

Radar chart
***********

Preliminaries: instantiation and execution of the MDO scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's start with the following code lines which instantiate and execute the :class:`~gemseo.core.mdo_scenario.MDOScenario` :

.. code::

   from gemseo.api import create_discipline, create_scenario

   formulation = 'MDF'

   disciplines = create_discipline(["SobieskiPropulsion", "SobieskiAerodynamics",
                                    "SobieskiMission", "SobieskiStructure"])

   scenario = create_scenario(disciplines,
                              formulation=formulation,
                              objective_name="y_4",
                              maximize_objective=True,
                              design_space="design_space.txt")

   scenario.set_differentiation_method("user")

   algo_options = {'max_iter': 10, 'algo': "SLSQP"}
   for constraint in ["g_1","g_2","g_3"]:
       scenario.add_constraint(constraint, 'ineq')

   scenario.execute(algo_options)

RadarChart
~~~~~~~~~~

Description
-----------

The :class:`~gemseo.post.radar_chart.RadarChart` post processing
plots on radar style chart a list of constraint functions
at a given iteration.

By default, the iteration is the last one.
It is possible either to save the plot, to show the plot or both.

This plot scales better with the number of constraints than the constraint plot provided by the :ref:`opt_history_view`

Options
-------

- **constraints_list**, :code:`list(str)` - list of constraints names
- **extension**, :code:`str` - file extension
- **file_path**, :code:`str` - the base paths of the files to export
- **iteration**, :code:`int` - number of iteration to post process
- **save**, :code:`bool` - if True, exports plot to pdf
- **show**, :code:`bool` - if True, displays the plot windows

Similarly, a radar constraint plot that scales better with the number of constraints, but at a single iteration,
can be plotted by using the API method :meth:`~gemseo.api.execute_post` with the keyword :code:`“RadarChart”` and
additional arguments concerning the type of display (file, screen, both).

Case of the MDF formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To visualize the radar constraint chart of the :code:`scenario`,
we use the :meth:`~gemseo.api.execute_post` API method with the keyword :code:`"ConstraintsHistory"`
and additional arguments concerning the type of display (file, screen, both):

.. code::

    scenario.post_process(“RadarChart”, constraints_list=["g_1", "g_2", "g_3"], save=True, show=False, file_path=“mdf”)

.. figure:: /_images/postprocessing/mdf_radar_chart.png
    :scale: 50 %

    History of the constraints on the Sobieski use case for the MDF
    formulation using the RadarChart plot
