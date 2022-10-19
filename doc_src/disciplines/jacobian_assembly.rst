..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Damien Guenot, Charlie Vanaret, Francois Gallard

.. _jacobian_assembly:

Coupled derivatives computation
-------------------------------

Reminder on adjoint method for gradient computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use of gradient-based methods implies the computation of the total derivatives of the output function
:math:`\bf{f}=(f_0,\ldots,f_N)^T` with respect to the design vector
:math:`\bf{x}=(x_0,\ldots,x_n)^T`:

 .. math::
    \frac{d\bf{f}}{d\bf{x}}=\begin{pmatrix}
    \displaystyle\frac{df_0}{d x_0} &\ldots&\displaystyle\frac{df_0}{dx_n}\\
    \vdots&\ddots&\vdots\\
    \displaystyle\frac{df_N}{d x_0} &\ldots&\displaystyle\frac{df_N}{dx_n}.
    \end{pmatrix}

A new feature of v1.0.0 of |g| is the management of gradients. Compared to v0.1.0, for which only finite differences or
complex step methods were available, jacobian assembly allows time savings and higher precision :ref:`MDAs<mda>` have to be solved for each
perturbed point :math:`(\bf{x}+h_j\bf{e}_j))`:

 .. math::
    \frac{d f_i}{d x_j} =
    \frac{f_i(\bf{x}+h_j\bf{e}_j)-f_i(\bf{x})}{h_j}+\mathcal{O}(h_j).

If the size of the design vector is large, it becomes very long to get the sensitivity of the output :math:`\bf{f}` with
respect to the design vector :math:`\bf{x}`.

Jacobian assembly is based on discrete adjoint theory ():

-  direct method: linear solve of :math:`\dfrac{d\bf{\mathcal{W}}}{d\bf{x}}`

   .. math::
      \dfrac{d\bf{f}}{d\bf{x}} = -\dfrac{\partial
      \bf{f}}{\partial \bf{\mathcal{W}}} \cdot \underbrace{\left[
      \left(\dfrac{\partial\bf{\mathcal{R}}}{\partial \bf{\mathcal{W}}}\right)^{-1}\cdot
      \dfrac{\partial \bf{\mathcal{R}}}{\partial \bf{x}}\right]}_{-d\bf{\mathcal{W}}/d\bf{x}}
      + \dfrac{\partial \bf{f}}{\partial \bf{x}}

-  adjoint method: computation of the adjoint vector :math:`\bf{\lambda}`

   .. math::

      \dfrac{d\bf{f}}{d\bf{x}} =
      -\underbrace{
      \left[ \dfrac{\partial \bf{f}}{\partial \bf{\mathcal{W}}} \cdot
      \left(\dfrac{\partial\bf{\mathcal{R}}}{\partial \bf{\mathcal{W}}}\right)^{-1} \right]}_{\bf{\lambda}^T} \cdot
      \dfrac{\partial \bf{\mathcal{R}}}{\partial \bf{x}}
      + \dfrac{\partial \bf{f}}{\partial \bf{x}} = -\bf{\lambda}^T\cdot
      \dfrac{\partial \bf{\mathcal{R}}}{\partial \bf{x}} + \dfrac{\partial \bf{f}}{\partial \bf{x}}

Dependency to design variable vector :math:`\bf{x}` has been removed.

The choice of which method (direct or adjoint) should be used depends on
how the number :math:`n` of outputs compares to the size of vector :math:`N`: if
:math:`N \ll n`, the adjoint method should be used, whereas the direct method
should be preferred if :math:`n\ll N`.

Both the direct and adjoint methods are implemented since |g| v1.0.0.

Derivatives computation in |g|: design and classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In |g|, the :class:`~gemseo.core.jacobian_assembly.JacobianAssembly` class computes the derivatives of the :ref:`MDAs<mda>`.
All :ref:`MDA<mda>` classes delegate the coupled derivatives computations to a :class:`~gemseo.core.jacobian_assembly.JacobianAssembly` instance.
The :class:`~gemseo.core.coupling_structure.MDOCouplingStructure` class is responsible for the analysis of the dependencies between the :class:`~gemseo.core.discipline.MDODiscipline`'s inputs and outputs, using a graph.


Many :ref:`MDA<mda>` algorithms are implemented in |g| (Gauss-Seidel, Jacobi, Newton variants).

.. uml::

   @startuml
   class MDODiscipline {
   +execute()
   }
   class MDA {
     +disciplines
     +jacobian_assembly
     +coupling_structure
     +_run()
   }
   class CouplingStructure {
     -_disciplines
     +weak_couplings()
     +strong_couplings()
     +weakly_coupled_disciplines()
     +strongly_coupled_disciplines()

   }
   class JacobianAssembly {
     -_coupling_structure
     +coupled_derivatives()
   }

      MDODiscipline <|-- MDA
      MDA "1" *-- "1" CouplingStructure
      MDA "1" *-- "1" JacobianAssembly
      MDA "1" -- "1..*" MDODiscipline
      JacobianAssembly "1" -- "1" CouplingStructure

   @end uml


Jacobian assembly: application to Sobieski's test-case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In |g|, the jacobian matrix of a discipline is a dictionary of dictionaries.
When wrapping the execution, a :meth:`!MDODiscipline._compute_jacobian` method must be
defined (it overloads the generical one defined in :class:`.MDODiscipline` class):
the jacobian matrix must be defined as :attr:`!MDODiscipline.jac`.

.. code::

    def _compute_jacobian(self, inputs=None, outputs=None, mode='auto'):
        """
        Compute the partial derivatives of all outputs wrt all inputs
        """
        # Initialize all matrices to zeros
        data_names = ["y_14", "y_24", "y_34", "x_shared"]
        y_14, y_24, y_34, x_shared = self.get_inputs_by_name(data_names)
        self.jac = self.sobieski_problem.derive_blackbox_mission(x_shared,
                                                                 y_14, y_24,
                                                                 y_34)


The differentiation method is set by the method :meth:`~gemseo.core.scenario.Scenario.set_differentiation_method` of :class:`~gemseo.core.scenario.Scenario`:

- for :code:`"finite_differences"` (default value):

.. code::

    scenario.set_differentiation_method("finite_differences")

- for the :code:`"complex_step"` method (each discipline must handle complex numbers):

.. code::

    scenario.set_differentiation_method("complex_step")

- for linearized version of the disciplines (:code:`"user"`): switching from direct mode to reverse mode is automatic, depending on the number of inputs and outputs. It can also be set by the user, setting :attr:`~gemseo.core.discipline.MDODiscipline.linearization_mode` at :code:`"direct"` or :code:`"adjoint"`).

.. code::

    scenario.set_differentiation_method("user")
    for discipline in scenario.disciplines:
       discipline.linearization_mode='auto' # default, can also be 'direct' or 'adjoint'


When deriving a tool, it is very easy to make some errors or to forget to derive some terms: that is why implementation of derivation can be validated
against finite differences or complex step method, by means of the method :meth:`~gemseo.core.discipline.MDODiscipline.check_jacobian`:

.. code::

    from gemseo.problems.sobieski.disciplines import SobieskiMission
    from gemseo.problems.sobieski.core import SobieskiProblem

    problem = SobieskiProblem("complex128")
    sr = SobieskiMission("complex128")
    sr.check_jacobian(indata, threshold=1e-12)

In order to be relevant, :code:`threshold` value should be kept at a low level
(:math:`<10^{-10}`).
