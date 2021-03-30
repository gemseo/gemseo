..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _opt_history_view:

History of evaluations
**********************

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

OptHistoryView
~~~~~~~~~~~~~~

Description
-----------

The **OptHistoryView** post processing
performs separated plots:
the design variables history,
the objective function history,
the history of hessian approximation of the objective,
the inequality constraint history,
the equality constraint history,
and constraints histories.

By default, all design variables are considered.
A sublist of design variables can be passed as options.
Minimum and maximum values for the plot can be passed as options.
The objective function can also be represented in terms of difference
w.r.t. the initial value
It is possible either to save the plot, to show the plot or both.

Options
-------

- **extension**, :code:`str` - file extension
- **file_path**, :code:`str` - the base paths of the files to export
- **obj_max**, :code:`float` - maximum value for the objective in the plot
- **obj_min**, :code:`float` - minimum value for the objective in the plot
- **obj_relative**, :code:`bool` - plot the objective value difference with the initial value
- **save**, :code:`bool` - if True, exports plot to pdf
- **show**, :code:`bool` - if True, displays the plot windows
- **variables_names**, :code:`list(str)` - list of the names of the variables to display

This post-processing feature is illustrated on the :code:`scenario`.

Case of the MDF formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To visualize the :term:`optimization history` of the :code:`scenario`,
we use the :meth:`~gemseo.api.execute_post` API method with the keyword :code:`"OptHistoryView"`
and additional arguments concerning the type of display (file, screen, both):

.. code::

    scenario.post_process("OptHistoryView", save=True, show=False,
                          file_path="mdf", extension="png")

This triggers the creation of five plots.

- **mdf_variables_history.png** - The first graph shows the normalized values of the design variables, the :math:`y` axis
  is the index of the inputs in the vector; and the :math:`x` axis represents the iterations.

.. _fig-ssbj-mdf-objj:

  .. figure:: /_images/postprocessing/mdf_variables_history.png
     :scale: 50 %

     Design variables history on the Sobieski use case for the MDF formulation

- **mdf_obj_history.png** - The second graph shows the evolution of the objective value during the optimization.

  .. figure:: /_images/postprocessing/mdf_obj_history.png
     :scale: 50 %

     Objective function history on the Sobieski use case for the MDF formulation

- **mdf_x_xstar_history.png** - The third graph plots the distance to the best design variables vector in log scale :math:`log( ||x-x^*|| )`.

  .. figure:: /_images/postprocessing/mdf_x_xstar_history.png
     :scale: 50 %

     Distance to the optimum history on the Sobieski use case for the MDF formulation

- **mdf_hessian_approx.png** - The fourth graph shows an approximation of the second order derivatives of the objective function
  :math:`\frac{\partial^2 f(x)}{\partial x^2}`, which is a measure of
  the sensitivity of the function with respect to the design variables,
  and of the anisotropy of the problem (differences of curvatures in the
  design space).

  .. figure:: /_images/postprocessing/mdf_hessian_approx.png
     :scale: 50 %

     Hessian diagonal approximation history on the Sobieski use case for the MDF formulation

- **mdf_ineq_constraints_history.png** - The last graph portrays the evolution of the values of the :term:`constraints`. The components of
  :math:`g\_1, g\_2, g\_3` are concatenated as a single vector of size 12.
  The constraints must be non-positive, that is the plot must be green or
  white for satisfied (white = active, red = violated). At convergence,
  only two are active.

.. _fig-ssbj-mdf-ineq:

  .. figure:: /_images/postprocessing/mdf_ineq_constraints_history.png
     :scale: 50 %

     History of the constraints on the Sobieski use case for the MDF formulation

Case of the IDF formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, if we consider the :ref:`IDF formulation <idf_formulation>`, after the execution of :code:`scenario`,
the optimization history may be visualized:

.. code::

    scenario.post_process("OptHistoryView", save=True, show=False,
                          file_path="idf", extension="png")

This triggers the creation of six plots.


- **idf_variables_history.png** - The first graph shows the normalized values of the design variables, the :math:`y` axis
  is the index of the inputs in the vector; and the :math:`x` axis represents the iterations. Compared to MDF, additional variables
  (indices 10 to 20), generated by the formulation, occur in the
  consistency .

  .. _fig-ssbj-idf-obj:
  .. figure:: /_images/postprocessing/idf_variables_history.png
     :scale: 50 %

     Design variables history on the Sobieski use case for the IDF formulation

- **idf_obj_history.png** - The second graph shows the evolution of the objective value during the optimization.

  .. figure:: /_images/postprocessing/idf_obj_history.png
     :scale: 50 %

     Objective function history on the Sobieski use case for the IDF formulation

- **idf_x_xstar_history.png** - The third graph plots the distance to the best design variables vector in log scale :math:`log( ||x-x^*|| )`.

  .. figure:: /_images/postprocessing/idf_x_xstar_history.png
     :scale: 50 %

     Distance to the optimum history on the Sobieski use case for the IDF formulation.

- **idf_hessian_approx.png** - The fourth graph shows an approximation of the second order derivatives of the objective function
  :math:`\frac{\partial^2 f(x)}{\partial x^2}`, which is a measure of
  the sensitivity of the function with respect to the design variables,
  and of the anisotropy of the problem (differences of curvatures in the
  design space).

  .. figure:: /_images/postprocessing/idf_hessian_approx.png
     :scale: 50 %

     Hessian diagonal approximation history on the Sobieski use case for the IDF formulation

- **mdf_ineq_constraints_history.png** - The last graphs portray the evolution of the values of the :term:`constraints`.
  Figures :ref:`fig-ssbj-idf-ineq` and :ref:`fig-ssbj-idf-eq` portray the evolution
  of the values of the inequality and equality constraints, respectively.
  The components of ``g_1, g_2, g_3`` are concatenated as a single vector
  of size 12. The inequality constraints must be non-positive, that is the
  plot must be green or white for satisfied constraints (white = active,
  red = violated). At convergence, only two inequality constraints are
  active.

  .. _fig-ssbj-idf-ineq:
  .. figure:: /_images/postprocessing/idf_ineq_constraints_history.png
     :scale: 50 %

     History of the inequality constraints on the Sobieski use case for
     the IDF formulation

  .. _fig-ssbj-idf-eq:
  .. figure:: /_images/postprocessing/idf_eq_constraints_history.png
     :scale: 50 %

     History of the equality constraints on the Sobieski use case for the
     IDF formulation
