..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Francois Gallard, Damien Guénot, Charlie Vanaret

.. _mda:

Multi Disciplinary Analyses
===========================

This section deals with the construction and execution of a Multi Disciplinary Analysis (MDA),
i.e. the computation and the convergence of the :term:`coupling variables` of coupled disciplines.

.. seealso::

   `The examples about MDA algorithms. <../examples/mda/index.html>`__

Creation of the MDAs
--------------------

Two families of MDA methods are implemented in |g|
and can be created with the function :func:`.create_mda`:

- :term:`fixed-point` algorithms,
    that compute fixed points of a coupling function :math:`y`,
    that is they solve the system :math:`y(x) = x`
- :term:`Newton method` algorithms for :term:`root finding`,
    that solve a non-linear multivariate problem :math:`R(x, y) = 0`
    using the derivatives :math:`\frac{\partial R(x, y)}{\partial y} = 0`
    where :math:`R(x, y)` is the residual.

Any explicit problem of the form :math:`y(x) = x`
can be reformulated into a root finding problem
by stating :math:`R(x, y) = y(x) - x = 0`.
The opposite is not true.
This means that if the disciplines are provided in a residual form,
the `Gauss-Seidel algorithm <https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method>`__
for instance cannot be directly used.

.. warning::

    Any :class:`.MDODiscipline` placed in an :class:`.MDA`
    with strong couplings **must** define its default inputs.
    Otherwise, the execution will fail.

Fixed-point algorithms
~~~~~~~~~~~~~~~~~~~~~~

.. _jacobi_method:

Fixed-point algorithms include
`Gauss-Seidel algorithm <https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method>`__
and `Jacobi <https://en.wikipedia.org/wiki/Jacobi_method>`__ algorithms.

.. code::

     from gemseo.api import create_mda, create_discipline

     disciplines = create_discipline(["SobieskiPropulsion", "SobieskiAerodynamics",
                                     "SobieskiMission", "SobieskiStructure"])

     mda_gaussseidel = create_mda("MDAGaussSeidel", disciplines)
     mda_jacobi = create_mda("MDAJacobi", disciplines)

.. seealso::

   The classes :class:`.MDAGaussSeidel`
   and :class:`.MDAJacobi`
   called by :func:`.create_mda`,
   inherit from :class:`.MDA`,
   itself inheriting from :class:`.MDODiscipline`.
   Therefore,
   an MDA based on a fixed-point algorithm can be viewed as a discipline
   whose inputs are design variables and outputs are coupling variables.

Root finding methods
~~~~~~~~~~~~~~~~~~~~

.. _newtonraphson_method:

Newton-Raphson method
^^^^^^^^^^^^^^^^^^^^^

The `Newton-Raphson method <https://en.wikipedia.org/wiki/Newton%27s_method>`__
is parameterized by a relaxation factor :math:`\alpha \in (0, 1]`
to limit the length of the steps taken along the Newton direction.
The new iterate is given by: :math:`x_{k+1} = x_k - \alpha f'(x_k)^{-1} f(x_k)`.

.. code::

    mda = create_mda("MDANewtonRaphson", disciplines, relax_factor=0.99)

Quasi-Newton method
^^^^^^^^^^^^^^^^^^^

`Quasi-Newton methods <https://en.wikipedia.org/wiki/Quasi-Newton_method>`__
include numerous variants (`Broyden <https://en.wikipedia.org/wiki/Broyden%27s_method>`__,
`Levenberg-Marquardt <https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`__, ...).
The name of the variant should be provided with the argument ``method``.

.. code::

    mda = create_mda("MDAQuasiNewton", disciplines, method=MDAQuasiNewton.BROYDEN1)

.. seealso::

   The classes :class:`.MDANewtonRaphson`
   and :class:`.MDAQuasiNewton`
   called by :func:`.create_mda`,
   inherit from :class:`.MDARoot`,
   itself inheriting from :class:`.MDA`,
   itself inheriting from :class:`.MDODiscipline`.
   Therefore,
   an MDA based on a root finding method can be viewed as a discipline
   whose inputs are design variables and outputs are coupling variables.

Hybrid methods
~~~~~~~~~~~~~~

Hybrid methods implement a generic scheme to combine elementary MDAs:
an arbitrary number of them are provided and are executed sequentially.
The following code creates a hybrid ``mda`` that runs sequentially
one iteration of :ref:`Jacobi method <jacobi_method>` ``mda1``
and a full :ref:`Newton-Raphson method <newtonraphson_method>` ``mda2``.

.. code::

    mda1 = create_mda("MDAJacobi", disciplines, max_mda_iter=1)
    mda2 = create_mda("MDANewtonRaphson", disciplines)
    mda = create_mda("MDASequential", disciplines, mda_sequence = [mda1, mda2])

This sequence is typically used to take advantage
of the robustness of fixed-point methods
and then obtain accurate results thanks to a Newton method.

Execution and convergence analysis
----------------------------------

The MDAs are run using the default input data
of the disciplines as a starting point.
A MDA provides a method to plot the evolution
of the residuals of the system with respect to the iterations ;
the plot may be displayed and/or saved with
:meth:`~MDA.plot_residual_history`:

.. code::

    mda.plot_residual_history(n_iterations=10, logscale=[1e-8, 10.])

The next plots compare the convergence of
Gauss-Seidel, Jacobi, quasi-Newton and the hybrid
with respect to the iterations.
Identical scales were used for the plots
(``n_iterations`` for the :math:`x` axis and ``logscale`` for the
logarithmic :math:`y` axis, respectively).
It shows that,
as expected,
Gauss-Seidel has a better convergence than the Jacobi method.
The hybrid MDA,
combining an iteration of Gauss-Seidel and a full Quasi-Newton,
converges must faster than all the other alternatives ;
note that Newton-Raphson alone does not converge well
for the initial values of the coupling variables.

.. figure:: /_images/mda/MDAGaussSeidel_residual_history.png
    :scale: 10 %

    Gauss-Seidel algorithm convergence for MDA.

.. figure:: /_images/mda/MDAJacobi_residual_history.png
    :scale: 10 %

    Jacobi algorithm convergence for MDA.

.. figure:: /_images/mda/MDAQuasiNewton_residual_history.png
    :scale: 10 %

    Quasi-Newton algorithm convergence for MDA.

.. figure:: /_images/mda/MDASequential_residual_history.png
    :scale: 10 %

    Hybrid Gauss-Seidel and a Quasi-Newton algorithm convergence for MDA.

Classes organization
--------------------

The following inheritance diagram shows the different MDA classes in |g| and their organization.

.. inheritance-diagram:: gemseo.mda.mda.MDA gemseo.mda.gauss_seidel.MDAGaussSeidel gemseo.mda.jacobi.MDAJacobi gemseo.mda.newton.MDANewtonRaphson gemseo.mda.sequential_mda.MDASequential gemseo.mda.sequential_mda.GSNewtonMDA gemseo.mda.newton.MDAQuasiNewton gemseo.mda.mda_chain.MDAChain
   :parts: 2


MDAChain and the Coupling structure for smart MDAs
--------------------------------------------------

The :class:`.MDOCouplingStructure`
provides methods to compute the coupling variables between the disciplines:

.. code::

    from gemseo.core.coupling_structure import MDOCouplingStructure

    coupling_structure = MDOCouplingStructure(disciplines)

This is an internal object that is created in all MDA classes and all formulations.
The end user does not need to create it for basic usage.

The :class:`.MDOCouplingStructure`
uses graphs to compute the dependencies between the disciplines,
and therefore the coupling variables.
This graph can then be used to generate a process
to solve the coupling problem with a coupling algorithm.

To illustrate the typical procedure,
we take a dummy 16 disciplines problem.

#. First the coupling graph is generated.
#. Then,
   a minimal process is computed,
   with eventually inner-MDAs.
   A set of coupling problems is generated,
   which are passed to algorithms.
#. Finally,
   a Jacobi MDA is used to solve the coupling equations,
   via the :term:`SciPy` package,
   or directly coded in |g| (Gauss-Seidel and Jacobi for instance).
   They can be compared on the specific problem,
   and MDAs can generate convergence plots of the residuals.

The next figure illustrates this typical process

.. figure:: /_images/mda/mda_auto_procedure.png
    :scale: 60 %

    The 3 resolution phases of a 16 disciplines coupling problem

This features is used in the :class:`.MDAChain`
which generates a chain of MDAs according
to the graph of dependency in order to minimize the execution time.
The user provides a base MDA class to solve the coupled problems.
The overall sequential process made of inner-MDAs and
disciplines execution is created by a :class:`.MDOChain`.
The inner-MDAs can be specified using the argument ``inner_mda_name``.

.. code::

    mda = create_mda("MDAChain", disciplines, inner_mda_name="MDAJacobi")
    mda.execute()
