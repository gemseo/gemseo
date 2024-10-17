..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Damien Guénot, Francois Gallard

:parenttoc: True
.. _post_processing:

How to deal with post-processing
================================

In this section we describe the post processing features of |g|, used to
analyze :class:`~gemseo.algos.optimization_result.OptimizationResult`, called the
:term:`optimization history`.

What data to post-process?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Post-processing features are applicable to any
:class:`~gemseo.algos.optimization_problem.OptimizationProblem` that has been solved,
which may have been loaded from the disk.

In practice,

- a :class:`~gemseo.scenarios.base_scenario.BaseScenario` instance has an :class:`~gemseo.formulations.base_mdo_formulation.BaseMDOFormulation` attribute,
- an :class:`~gemseo.formulations.base_mdo_formulation.BaseMDOFormulation` instance has an :class:`~gemseo.algos.optimization_problem.OptimizationProblem` attribute,
- an :class:`~gemseo.algos.optimization_problem.OptimizationProblem` instance has an :class:`~gemseo.algos.optimization_result.OptimizationResult` attribute.

Illustration on the Sobieski use case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The post-processing features are illustrated on MDO results obtained on the :ref:`SSBJ use case <sobieski_problem>`,
using different types of ``formulation`` (:ref:`MDF formulation <mdf_formulation>`, :ref:`IDF formulation <idf_formulation>`, ...)

The following code sets up and executes the problem. It is possible to try different types of MDO strategies by changing
the ``formulation`` value. For a detailed explanation on how to setup the case, please see
:ref:`sphx_glr_examples_mdo_plot_sobieski_use_case.py`.


.. code::

   from gemseo import create_discipline, create_scenario

   formulation = 'MDF'

   disciplines = create_discipline(["SobieskiPropulsion", "SobieskiAerodynamics",
                                    "SobieskiMission", "SobieskiStructure"])

   scenario = create_scenario(disciplines,
                              formulation=formulation,
                              objective_name="y_4",
                              maximize_objective=True,
                              design_space="design_space.csv")

   scenario.set_differentiation_method("user")

   for constraint in ["g_1","g_2","g_3"]:
       scenario.add_constraint(constraint, 'ineq')

   scenario.execute(algo_name="SLSQP", max_iter=10)

How to apply a post-process feature?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From this ``scenario``, we can apply any kind of post-processing dedicated to :class:`~gemseo.scenarios.base_scenario.BaseScenario` instances,

- either by means of its :meth:`~gemseo.scenarios.base_scenario.BaseScenario.post_process` method:

    .. automethod:: gemseo.scenarios.base_scenario.BaseScenario.post_process
       :noindex:

- or by means of the :func:`.execute_post` API method:

    .. autofunction:: gemseo.execute_post
       :noindex:

.. note::

    Only design variables and functions (objective function, constraints) are stored for post-processing.
    If you want to be able to plot state variables, you must add them as observables before the problem is executed.
    Use the :meth:`~gemseo.scenarios.base_scenario.BaseScenario.add_observable` method.

.. include:: /examples/post_process/index.rst
   :start-after: start-after-label
