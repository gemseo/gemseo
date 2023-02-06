..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Damien Guénot, Francois Gallard

.. include:: ../includes/big_toc_css.rst
.. include:: ../tune_toc.rst

.. _post_processing:

How to deal with post-processing
================================

In this section we describe the post processing features of |g|, used to
analyze :class:`~gemseo.algos.opt_result.OptimizationResult`, called the
:term:`optimization history`.

What data to post-process?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Post-processing features are applicable to any
:class:`~gemseo.algos.opt_problem.OptimizationProblem` that has been solved,
which may have been loaded from the disk.

In practice,

- a :class:`~gemseo.core.scenario.Scenario` instance has an :class:`~gemseo.core.formulation.MDOFormulation` attribute,
- an :class:`~gemseo.core.formulation.MDOFormulation` instance has an :class:`~gemseo.algos.opt_problem.OptimizationProblem` attribute,
- an :class:`~gemseo.algos.opt_problem.OptimizationProblem` instance has an :class:`~gemseo.algos.opt_result.OptimizationResult` attribute.

Illustration on the Sobieski use case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The post-processing features are illustrated on MDO results obtained on the :ref:`SSBJ use case <sobieski_problem>`,
using different types of :code:`formulation` (:ref:`MDF formulation <mdf_formulation>`, :ref:`IDF formulation <idf_formulation>`, ...)

The following code sets up and executes the problem. It is possible to try different types of MDO strategies by changing
the :code:`formulation` value. For a detailed explanation on how to setup the case, please see
:ref:`sphx_glr_examples_mdo_plot_sobieski_use_case.py`.


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

How to apply a post-process feature?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From this :code:`scenario`, we can apply any kind of post-processing dedicated to :class:`~gemseo.core.scenario.Scenario` instances,

- either by means of its :meth:`~gemseo.core.scenario.Scenario.post_process` method:

    .. automethod:: gemseo.core.scenario.Scenario.post_process
       :noindex:

- or by means of the :meth:`~gemseo.api.execute_post` API method:

    .. autofunction:: gemseo.api.execute_post
       :noindex:

.. note::

    Only design variables and functions (objective function, constraints) are stored for post-processing.
    If you want to be able to plot state variables, you must add them as observables before the problem is executed.
    Use the :meth:`~gemseo.core.scenario.Scenario.add_observable` method.

What are the post-processing features?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <h3><a href="../examples/post_process/plot_basic_history.html">Basic history</a></h3>
   <a href="../examples/post_process/plot_basic_history.html"><img src="../_images/sphx_glr_plot_basic_history_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_constraints_history.html">Constraints history</a></h3>
   <a href="../examples/post_process/plot_constraints_history.html"><img src="../_images/sphx_glr_plot_constraints_history_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_obj_constr_hist.html">Objective and constraints history</a></h3>
   <a href="../examples/post_process/plot_obj_constr_hist.html"><img src="../_images/sphx_glr_plot_obj_constr_hist_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_gradient_sensitivity.html">Gradient sensitivity</a></h3>
   <a href="../examples/post_process/plot_gradient_sensitivity.html"><img src="../_images/sphx_glr_plot_gradient_sensitivity_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_opt_hist_view.html">Optimization history view</a></h3>
   <a href="../examples/post_process/plot_opt_hist_view.html"><img src="../_images/sphx_glr_plot_opt_hist_view_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_para_coord.html">Parallel coordinates</a></h3>
   <a href="../examples/post_process/plot_para_coord.html"><img src="../_images/sphx_glr_plot_para_coord_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_quad_approx.html">Quadratic approximation</a></h3>
   <a href="../examples/post_process/plot_quad_approx.html"><img src="../_images/sphx_glr_plot_quad_approx_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_constraints_radar_chart.html">Radar chart</a></h3>
   <a href="../examples/post_process/plot_constraints_radar_chart.html"><img src="../_images/sphx_glr_plot_constraints_radar_chart_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_correlations.html">Correlations</a></h3>
   <a href="../examples/post_process/plot_correlations.html"><img src="../_images/sphx_glr_plot_correlation_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_robustness.html">Robustness</a></h3>
   <a href="../examples/post_process/plot_robustness.html"><img src="../_images/sphx_glr_plot_robustness_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_som.html">Self-Organizing Maps</a></h3>
   <a href="../examples/post_process/plot_som.html"><img src="../_images/som_fine.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_history_scatter_matrix.html">Scatter plot matrix</a></h3>
   <a href="../examples/post_process/plot_history_scatter_matrix.html"><img src="../_images/sphx_glr_plot_history_scatter_matrix_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_variable_influence.html">Variable influence</a></h3>
   <a href="../examples/post_process/plot_variable_influence.html"><img src="../_images/sphx_glr_plot_variable_influence_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>
   <h3><a href="../examples/post_process/plot_pareto_front.html">Pareto front</a></h3>
   <a href="../examples/post_process/plot_pareto_front.html"><img src="../_images/sphx_glr_plot_pareto_front_001.png" style="height:75px; margin-right:5px;" align="center"/></a><br/>

.. toctree::
   :caption: Post processing methods
   :maxdepth: 1
   :hidden:

   /examples/post_process/plot_basic_history
   /examples/post_process/plot_constraints_history
   /examples/post_process/plot_obj_constr_hist
   /examples/post_process/plot_gradient_sensitivity
   /examples/post_process/plot_opt_hist_view
   /examples/post_process/plot_para_coord
   /examples/post_process/plot_quad_approx
   /examples/post_process/plot_constraints_radar_chart
   /examples/post_process/plot_correlations
   /examples/post_process/plot_robustness
   /examples/post_process/plot_som
   /examples/post_process/plot_history_scatter_matrix
   /examples/post_process/plot_variable_influence
   /examples/post_process/plot_pareto_front
