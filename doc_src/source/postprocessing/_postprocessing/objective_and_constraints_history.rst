..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _constraints_history:

Objective and constraints history
*********************************

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

ObjConstrHist
~~~~~~~~~~~~~

Description
-----------

The :class:`~gemseo.post.obj_constr_hist.ObjConstrHist` post processing
plots the objective history in a line chart
with constraint violation indication by color in the background.

By default, all the constraints are considered. A sublist of constraints
can be passed as options.
It is possible either to save the plot, to show the plot or both.

Options
-------

- **constr_names**, :code:`list(str)` - The constraint names to plot. If empty, plot all.
- **extension**, :code:`str` - The file extension.
- **file_path**, :code:`str` - The base paths of the files to export. Relative to the working directory.
- **save**, :code:`bool` - If True, export plot to a file.
- **show**, :code:`bool` - If True, display the plot windows.

Case of the MDF formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To visualize the objective and constraints history of the :code:`scenario`,
we use the :meth:`~gemseo.api.execute_post` API method with the keyword :code:`"ObjConstrHist"`
and additional arguments concerning the type of display (file
, screen, both):

.. code::

    scenario.post_process("ObjConstrHist", constr_names=["g_1", "g_2", "g_3"], save=True, show=False, file_path="mdf")


.. figure:: /_images/postprocessing/mdf_objective_and_constraints_history.png
    :scale: 70 %

    History of the constraints on the Sobieski use case for the MDF
    formulation using the ObjConstrHist plot.
