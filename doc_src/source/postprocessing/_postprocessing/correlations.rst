..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _correlations:

Correlations
************

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

Correlations
~~~~~~~~~~~~

Description
-----------

A correlation coefficient indicates whether there is a linear
relationship between 2 quantities :math:`x` and :math:`y`, in which case
it equals 1 or -1. It is the normalized covariance between the two
quantities:

.. math::

   R_{xy}=\frac {\sum \limits _{i=1}^n(x_i-{\bar{x}})(y_i-{\bar{y}})}{ns_{x}s_{y}}=
   \frac {\sum \limits _{i=1}^n(x_i-{\bar{x}})(y_i-{\bar{y}})}{\sqrt {\sum
   \limits _{i=1}^n(x_i-{\bar{x}})^{2}\sum \limits _{i=1}^n(y_i-{\bar{y}})^{2}}}

The **Correlations** post processing builds scatter plots of correlated variables among
design variables, output functions, and constraints.

The plot method considers all variable correlations greater than 95%. A different
threshold value and/or a sublist of variable names can be passed as options. The x-
and y-figure sizes can also be modified in options. It is possible to either save
the plot, to show the plot or both.

Options
-------

- **func_names**, :code:`list(str)` - The func_names for which the correlations are computed. If None, all functions are considered.
- **coeff_limit**, :code:`float` - If the correlation between the variables is lower than coeff_limit, the plot is not made.
- **n_plots_x**, :code:`int` - The number of horizontal plots.
- **n_plots_y**, :code:`int` - The number of vertical plots.
- **save**, :code:`bool` - If True, export the plot to file.
- **show**, :code:`bool` - If True, display the plot windows.
- **file_path**, :code:`str` - The base paths of the files to export. If None, use the current working directory.
- **extension**, :code:`str` - The file extension.
- **figsize_x**, :code:`int` - The size of the figure in the horizontal direction (inches).
- **figsize_y**, :code:`int` - The size of figure in the vertical direction (inches).

Case of the MDF formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compute the correlations between all inputs and all outputs as well
as between two outputs, use the API method :meth:`~gemseo.api.execute_post`
with the keyword :code:`"Correlations"` and
additional arguments concerning the type of display (file, screen, both):

.. code::

    scenario.post_process("Correlations", coeff_limit=0.85, save=True,
    show=False, n_plots_x=4, n_plots_y=4)

where:

-  ``coeff_limit`` is the absolute threshold for correlation plots. It
   filters the minimum correlation coefficient to be displayed,

-  ``n_plot_x`` and ``n_plot_y`` are the numbers of plots along the
   columns and the rows, respectively.

.. figure:: /_images/postprocessing/mdf_correlations.png
   :alt: Correlation coefficients on the Sobieski use case for the MDF formulation
   :scale: 50 %

   Correlation coefficients on the Sobieski use case for the MDF
   formulation

As mentioned earlier, correlation plots highlight the strong correlations between stress
constraints in wing sections : the correlation coefficients belong to
:math:`[0.94766, 0.999286]`.

The aerodynamics constraint ``g_2`` is a polynomial function of
:math:`x_1`: :math:`g\_2=1+0.2\overline{x_1}` with
:math:`\overline{x_1}` the normalized value of :math:`x_1`.
