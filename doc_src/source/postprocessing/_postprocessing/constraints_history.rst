..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _constraints_history:

Constraints history
*******************

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

ConstraintsHistory
~~~~~~~~~~~~~~~~~~

Description
-----------

The :class:`~gemseo.post.constraints_history.ConstraintsHistory` post processing
plots the constraints functions history in lines charts
with violation indication by color on background.

The plot method requires the list of constraint names to plot.
It is possible either to save the plot, to show the plot or both.

This plot is more precise than the constraint plot provided by the :ref:`opt_history_view`
but scales less with the number of constraints

Options
-------

- **constraints_list**, :code:`list(str)` - list of constraint names
- **extension**, :code:`str` - file extension
- **file_path**, :code:`str` - the base paths of the files to export
- **save**, :code:`bool` - if True, exports plot to pdf
- **show**, :code:`bool` - if True, displays the plot windows

Case of the MDF formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To visualize the constraints history of the :code:`scenario`,
we use the :meth:`~gemseo.api.execute_post` API method with the keyword :code:`"ConstraintsHistory"`
and additional arguments concerning the type of display (file
, screen, both):

.. code::

    scenario.post_process("ConstraintsHistory", constraints_list=["g_1", "g_2", "g_3"], save=True, show=False, file_path="mdf")


.. figure:: /_images/postprocessing/mdf_constraints_history.png
    :scale: 50 %

    History of the constraints on the Sobieski use case for the MDF
    formulation using the ConstraintsHistory plot
