..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _parallel_coordinates:

Parallel coordinates
********************

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

ParallelCoordinates
~~~~~~~~~~~~~~~~~~~

Description
-----------

The :class:`~gemseo.post.para_coord.ParallelCoordinates` post processing
builds parallel coordinates plots  among design
variables, outputs functions and constraints

The :class:`~gemseo.post.para_coord.ParallelCoordinates` portray the design variables history during the
scenario execution. Each vertical coordinate is dedicated to a design
variable, normalized by its bounds.

A polyline joins all components of a given design vector and is colored
by objective function values. This highlights correlations between
values of the design variables and values of the objective function.

x- and y- figure sizes can be changed in option.
It is possible either to save the plot, to show the plot or both.

Options
-------

- **extension**, :code:`str` - file extension
- **figsize_x**, :code:`Unknown` - X size of the figure Default value = 10
- **figsize_y**, :code:`Unknown` - Y size of the figure Default value = 2
- **file_path**, :code:`str` - the base paths of the files to export
- **save**, :code:`bool` - if True, exports plot to pdf
- **show**, :code:`bool` - if True, displays the plot windows

Case of the MDF formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To plot the parallel coordinates, use the API method :meth:`~gemseo.api.execute_post`
with the keyword :code:`"ParallelCoordinates"` and
additional arguments concerning the type of display (file, screen, both):

.. code::

    scenario.post_process("ParallelCoordinates", save=True, show=False, file_path="mdf" )

Figures :ref:`fig-ssbj-mdf-quadapprox` and :ref:`fig-ssbj-mdf-quadapprox`
show parallel coordinates respectively of the :term:`design variables`
and the :term:`constraints`, colored by :term:`objective function` value. The
minimum objective value, in dark blue, corresponds to satisfied
constraints.

.. _fig-ssbj-mdf-paracoords-desvar:

.. figure:: /_images/postprocessing/mdf_para_coord_des_vars.png
    :scale: 50 %

    Parallel coordinates of design variables on the Sobieski use case for
    the MDF formulation

.. _fig-ssbj-mdf-paracoords-funcs:

.. figure:: /_images/postprocessing/mdf_para_coord_funcs.png
    :scale: 50 %

    Parallel coordinates of constraints on the Sobieski use case for the
    MDF formulation
