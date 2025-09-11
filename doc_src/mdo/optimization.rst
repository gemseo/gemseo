..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Francois Gallard

.. _optimization:

Optimization and DOE framework
==============================

In this section we describe |g|'s optimization and DOE framework.

The standard way to use |g| is through an :class:`.MDOScenario`, which
automatically creates an :class:`.OptimizationProblem` from an :ref:`MDO formulation <mdo_formulations>` and a set of
:class:`.Discipline`.

However, one may be interested in directly creating an :class:`.OptimizationProblem` using the class :class:`.OptimizationProblem`,
which can be solved using an :term:`optimization algorithm` or sampled with a :term:`DOE algorithm`.

.. warning::

   :ref:`MDO formulation <mdo_formulations>` and optimization problem developers should also understand this part of |g|.


Setting up an :class:`.OptimizationProblem`
-------------------------------------------

The :class:`.OptimizationProblem` class is composed of at least a
:class:`.DesignSpace` created from :func:`.create_design_space` which describes the :term:`design variables`:

.. code:: python

    from gemseo import create_design_space
    from numpy import ones

    design_space = create_design_space()
    design_space.add_variable("x", 1, lower_bound=-2., upper_bound=2.,
                              value=-0.5 * ones(1))

and an objective function, of type :class:`.MDOFunction`. The :class:`.MDOFunction` is callable and requires at least
a function pointer to be instantiated. It supports expressions and the +, -, \ * operators:

.. code:: python

    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from numpy import cos
    from numpy import exp
    from numpy import sin

    f_1 = MDOFunction(sin, name="f_1", jac=cos, expr="sin(x)")
    f_2 = MDOFunction(exp, name="f_2", jac=exp, expr="exp(x)")
    f_1_sub_f_2 = f_1 - f_2

From this :class:`.DesignSpace`, an :class:`.OptimizationProblem` is built:

.. code:: python

    from gemseo.algos.optimization_problem import OptimizationProblem
    problem = OptimizationProblem(design_space)

To set the objective :class:`.MDOFunction`, the attribute :attr:`.OptimizationProblem.objective` of class :class:`.OptimizationProblem`
must be set with the objective function pointer:

.. code:: python

   problem.objective = f_1_sub_f_2

Similarly the :attr:`.OptimizationProblem.constraints` attribute must be set with a list of inequality or equality constraints.
The :attr:`.MDOFunction.f_type` attribute of :class:`.MDOFunction` shall be set to ``"eq"`` or ``"ineq"`` to declare the type of constraint to equality or inequality.

.. warning::

   **All inequality constraints must be negative by convention**, whatever the optimization algorithm used to solve the problem.

Solving the problem by optimization
-----------------------------------

Once the optimization problem created, it can be solved using one of the available
optimization algorithms from the :class:`.OptimizationLibraryFactory`,
by means of the function :meth:`.BaseOptimizationLibrary.execute`
whose mandatory arguments are the :class:`.OptimizationProblem`
and the optimization algorithm name. For example, in the case of the `L-BFGS-B algorithm <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_
with normalized design space, we have:

.. code:: python

    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    opt = OptimizationLibraryFactory().execute(problem, algo_name="L-BFGS-B",
                                      normalize_design_space=True)
    print(f"Optimum = {opt.f_opt}")

Note that the `L-BFGS-B algorithm <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_ is implemented in the external
library `SciPy <https://scipy.org/>`_
and interfaced with |g| through the class :class:`.ScipyOpt`.

The list of available algorithms depend on the local setup of |g|, and the installed
optimization libraries. It can be obtained using :

.. code:: python

    algo_list = OptimizationLibraryFactory().algorithms
    print(f"Available algorithms: {algo_list}")

The optimization history can be saved to the disk for further analysis,
without having to re execute the optimization.
For that, we use the function :meth:`.OptimizationProblem.to_hdf`:

.. code:: python

    problem.to_hdf("simple_opt.hdf5")

Solving the problem by DOE
--------------------------

:term:`DOE` algorithms can also be used to sample the design space and observe the
value of the objective and constraints

.. code:: python

    from gemseo.algos.doe.factory import DOELibraryFactory

    # And solve it with |g| interface
    opt = DOELibraryFactory().execute(
        problem, algo_name="PYDOE_LHS", n_samples=10, normalize_design_space=True
    )

Results analysis
----------------

The optimization history can be plotted using one of the post processing tools, see the :ref:`post-processing <post_processing>` page.

.. code:: python

    from gemseo import execute_post

    execute_post(problem, post_name="OptHistoryView", save=True, file_path="simple_opt")

    # Also works from disk
    execute_post("my_optim.hdf5", post_name="OptHistoryView", save=True, file_path="opt_view_from_disk")

.. _fig-ssbj-mdf-obj:

.. figure:: /_images/doe/simple_opt.png
    :scale: 50 %

    Objective function history for the simple analytic optimization


.. _doe_algos:

DOE algorithms
--------------

|g| is interfaced with two packages that provide DOE algorithms:
`pyDOE <https://pythonhosted.org/pyDOE/>`_, and
`OpenTURNS <https://openturns.github.io/www/>`_.
To list the available DOE algorithms in the current |g| configuration, use
:func:`.get_available_doe_algorithms`.

The set of plots below shows plots using various available algorithms.


.. figure::  /_images/doe/fullfact_pyDOE.png
   :scale: 40%

   Full factorial DOE from pyDOE


.. figure::  /_images/doe/bbdesign_pyDOE.png
   :scale: 40%

   Box-Behnken DOE from pyDOE


.. figure:: /_images/doe/lhs_pyDOE.png
   :scale: 40%

   LHS DOE from pyDOE

.. figure::  /_images/doe/axial_openturns.png
   :scale: 40%

   Axial DOE from OpenTURNS

.. figure:: /_images/doe/composite_openturns.png
   :scale: 40%

   Composite DOE from OpenTURNS

.. figure:: /_images/doe/factorial_openturns.png
   :scale: 40%

   Full Factorial DOE from OpenTURNS

.. figure::  /_images/doe/faure_openturns.png
   :scale: 40%

   Faure DOE from OpenTURNS

.. figure:: /_images/doe/halton_openturns.png
   :scale: 40%

   Halton DOE from OpenTURNS

.. figure:: /_images/doe/haselgrove_openturns.png
   :scale: 40%

   Haselgrove DOE from OpenTURNS

.. figure::  /_images/doe/sobol_openturns.png
   :scale: 40%

   Sobol DOE from OpenTURNS

.. figure::  /_images/doe/mc_openturns.png
   :scale: 40%

   Monte-Carlo DOE from OpenTURNS

.. figure::  /_images/doe/lhsc_openturns.png
   :scale: 40%

   LHSC DOE from OpenTURNS

.. figure::  /_images/doe/lhs_openturns.png
   :scale: 40%

   LHS DOE from OpenTURNS

.. figure::  /_images/doe/random_openturns.png
   :scale: 40%

   Random DOE from OpenTURNS
