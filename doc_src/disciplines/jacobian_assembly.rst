..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Damien Guenot, Charlie Vanaret, Francois Gallard, Sebastien Bocquet

.. _jacobian_assembly:

Coupled derivatives and gradients computation
---------------------------------------------

Introduction
^^^^^^^^^^^^

The use of gradient-based methods implies the computation of the total derivatives,
or Jacobian matrix, of the objective function and constraints.

:math:`\bf{f}=(f_0,\ldots,f_N)^T` with respect to the design vector
:math:`\bf{x}=(x_0,\ldots,x_n)^T`:

 .. math::
    \frac{d\bf{f}}{d\bf{x}}=\begin{pmatrix}
    \displaystyle\frac{df_0}{d x_0} &\ldots&\displaystyle\frac{df_0}{dx_n}\\
    \vdots&\ddots&\vdots\\
    \displaystyle\frac{df_N}{d x_0} &\ldots&\displaystyle\frac{df_N}{dx_n}.
    \end{pmatrix}

The Jacobian matrix may be either approximated (by the finite differences and
complex step), or computed analytically. Both options are possible in |g|.

The analytic computation of the derivatives is usually a better approach,
since it is cheaper and more precise than the approximations.
The tuning of the finite difference step is difficult, although some methods
exist for that (:meth:`~gemseo.core.discipline.Discipline.set_optimal_fd_step`), and
the complete MDO process has to be evaluated for each
perturbed point :math:`(\bf{x}+h_j\bf{e}_j))`, which scales badly with
the number of design variables.

 .. math::
    \frac{d f_i}{d x_j} =
    \frac{f_i(\bf{x}+h_j\bf{e}_j)-f_i(\bf{x})}{h_j}+\mathcal{O}(h_j).


As a result it is crucial to select an appropriate method to compute the gradient.

However, the computation of analytic derivatives is not straightforward.
It requires to compute the jacobian of the objective function and constraints,
that are computed by a |g| process (such as a chain or a MDA), and from
the derivatives provided by all the disciplines.

For weakly coupled problems, based on :class:`~gemseo.core.chain.MDOChain`, the generalized
chain rule in reverse mode (from the outputs to the inputs) is used.

For the coupled problems, when the process is based on a MDA :class:`~gemseo.mda.BaseMDA`,
a coupled adjoint approach is used, with two
variants (direct or adjoint) depending on the number of design variables
compared to the number of objectives and constraints.

The following section describes the coupled adjoint method.

The coupled adjoint theory
^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following two strongly coupled disciplines problems.
The coupling variables  :math:`\mathcal{Y_1}` and :math:`\mathcal{Y_2}`,
are vectors computed by each discipline,
and depend on the output of the other discipline.
This is formalized as two equality constraints:

.. math::
  \left\{
      \begin{aligned}
        \mathcal{Y_1} = \mathcal{Y_1}(\mathbf{x}, \mathcal{Y_2})\\
        \mathcal{Y_2} = \mathcal{Y_2}(\mathbf{x}, \mathcal{Y_1})\\
      \end{aligned}
  \right.

where :math:`\mathbf{x}` may be a vector.

It can be rewritten in a residual form:

.. math::
  \left\{
      \begin{aligned}
        \mathcal{Y_1}(\mathbf{x}, \mathcal{Y_2}) - {\mathcal{Y_1}}^\intercal = 0\\
        \mathcal{Y_2}(\mathbf{x}, \mathcal{Y_1}) - {\mathcal{Y_2}}^\intercal = 0\\
      \end{aligned}
    \right.

Solving the MDA can be summarized as solving a vector residual equation:

.. math::
   \forall \mathbf{x}, \mathcal{R}(\mathbf{x}, \mathbf{\mathcal{Y}(x)}) = \mathbf{0}

with

.. math::
   \mathcal{Y} =
   \left\{
     \begin{aligned}
       {\mathcal{Y_1}}^\intercal\\
       {\mathcal{Y_2}}^\intercal\\
     \end{aligned}
   \right.

This assumes that the MDA is exactly solved at each iteration.

Since :math:`\mathcal{R}` is a null function, its derivative with respect
to :math:`\mathbf{x}` is always zero, leading to:

.. math::
   \frac{\partial \mathcal{R}}{\partial \mathbf{x}}
   + \frac{\partial \mathcal{R}}{\partial \mathcal{Y}}~
   \frac{d\mathcal{Y}}{d\mathbf{x}} = 0

So we can obtain the total derivative of the coupling variables :

.. math::
   \frac{d\mathcal{Y}}{d\mathbf{x}} =
   -\left( \frac{\partial \mathcal{R}}{\partial \mathcal{Y}} \right)^{-1}
   \frac{\partial \mathcal{R}}{\partial \mathbf{x}}

However, this linear system is very expensive when there are many design variables,
so the computation of this derivative is usually not performed.

So the objective function, that depends on both the coupling and
design variables :math:`\mathbf{f}(\mathbf{x}, \mathcal{Y}(\mathbf{x}))`, is derived:

.. math::
   \frac{d\mathbf{f}}{d\mathbf{x}} =
   \frac{\partial \mathbf{f}}{\partial \mathbf{x}} +
   \frac{\partial \mathbf{f}}{\partial \mathcal{Y}}~
   \frac{d\mathcal{Y}}{d\mathbf{x}}

Replacing :math:`\frac{d\mathcal{Y}}{d\mathbf{x}}` from the residual derivative gives:

.. math::
  :name: eq:f_gradient

   \frac{d\mathbf{f}}{d\mathbf{x}} =
   - \frac{\partial \mathbf{f}}{\partial \mathcal{Y}}~
   \left( \frac{\partial \mathcal{R}}{\partial \mathcal{Y}} \right)^{-1}~
   \frac{\partial \mathcal{R}}{\partial \mathbf{x}}
   + \frac{\partial \mathbf{f}}{\partial \mathbf{x}}


Adjoint versus direct methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cost of evaluating the gradient of :math:`\mathbf{f}` is driven by the matrix inversion
:math:`\left( \frac{\partial \mathcal{R}}{\partial \mathcal{Y}} \right)^{-1}`.
Two approaches are possible to compute the previous equation:

  -  The adjoint method: computation of the adjoint vector :math:`\bf{\lambda}`

     .. math::

        \dfrac{d\bf{f}}{d\bf{x}} =
        -\underbrace{
        \left[ \dfrac{\partial \bf{f}}{\partial \bf{\mathcal{Y}}} \cdot
        \left(\dfrac{\partial\bf{\mathcal{R}}}{\partial \bf{\mathcal{Y}}}\right)^{-1} \right]}_{\bf{\lambda}^T} \cdot
        \dfrac{\partial \bf{\mathcal{R}}}{\partial \bf{x}}
        + \dfrac{\partial \bf{f}}{\partial \bf{x}} = -\bf{\lambda}^T\cdot
        \dfrac{\partial \bf{\mathcal{R}}}{\partial \bf{x}} + \dfrac{\partial \bf{f}}{\partial \bf{x}}

     The adjoint vector is obtained by solving one linear system per output
     function (objective and constraint).

    .. math::

        \dfrac{\partial\bf{\mathcal{R}}}{\partial \bf{\mathcal{Y}}} ^T \lambda - \dfrac{\partial \bf{f}}{\partial \bf{\mathcal{Y}}}^T = 0

    These linear systems are the expensive part of the computation, which does not depend on
    the number of design variables because the equation is independent of x.
    The Jacobian of the functions are then obtained by a simple matrix vector product,
    which cost depends on the design variables number but is usually negligible.

  -  the direct method: linear solve of :math:`\dfrac{d\bf{\mathcal{Y}}}{d\bf{x}}`

     .. math::
        \dfrac{d\bf{f}}{d\bf{x}} = -\dfrac{\partial
        \bf{f}}{\partial \bf{\mathcal{Y}}} \cdot \underbrace{\left[
        \left(\dfrac{\partial\bf{\mathcal{R}}}{\partial \bf{\mathcal{Y}}}\right)^{-1}\cdot
        \dfrac{\partial \bf{\mathcal{R}}}{\partial \bf{x}}\right]}_{-d\bf{\mathcal{Y}}/d\bf{x}}
        + \dfrac{\partial \bf{f}}{\partial \bf{x}}

    The computational cost is driven by the linear systems, one per design variable.
    It does not depend on the number of output function, so is well adapted when there
    are more function outputs than design variables.


The choice of which method (direct or adjoint) should be used depends on
how the number :math:`n` of outputs compares to the size of vector :math:`N`: if
:math:`N \ll n`, the adjoint method should be used, whereas the direct method
should be preferred if :math:`n\ll N`.

Both the direct and adjoint methods are implemented since |g| v1.0.0, and the
switch between the direct or adjoint method is automatic, but can be forced by the user.

Object oriented design
^^^^^^^^^^^^^^^^^^^^^^

In |g|, the :class:`~gemseo.core.jacobian_assembly.JacobianAssembly` class computes the derivatives of the :ref:`MDAs<mda>`.
All :ref:`MDA<mda>` classes delegate the coupled derivatives computations to a
:class:`~gemseo.core.jacobian_assembly.JacobianAssembly` instance.
The :class:`~gemseo.core.coupling_structure.CouplingStructure` class is responsible for the analysis of the
dependencies between the :class:`~gemseo.core.discipline.Discipline`'s inputs and outputs, using a graph.


Many :ref:`MDA<mda>` algorithms are implemented in |g| (Gauss-Seidel, Jacobi, Newton variants).

.. uml::

   @startuml

   class Discipline {
   +execute()
   }
   class MDA {
     +disciplines
     +jacobian_assembly
     +coupling_structure
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

   Discipline <|-- MDA
   MDA "1" *-- "1" CouplingStructure
   MDA "1" *-- "1" JacobianAssembly
   MDA "1" -- "1..*" Discipline
   JacobianAssembly "1" -- "1" CouplingStructure

   @enduml


Illustration on the Sobieski SSBJ test-case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In |g|, the jacobian matrix of a discipline is a dictionary of dictionaries.
When wrapping the execution, a :meth:`!Discipline._compute_jacobian` method must be
defined (it overloads the generical one defined in :class:`.Discipline` class):
the jacobian matrix must be defined as :attr:`!Discipline.jac`.

.. code::

    def _compute_jacobian(self, input_names=(), output_names=()):
        """
        Compute the partial derivatives of all outputs wrt all inputs
        """
        y_14 = self.local_data["y_14"]
        y_24 = self.local_data["y_24"]
        y_34 = self.local_data["y_34"]
        x_shared = self.local_data["x_shared"]
        self.jac = self.sobieski_problem.derive_blackbox_mission(x_shared, y_14, y_24, y_34)

The differentiation method is set by the method :meth:`~gemseo.scenarios.base_scenario.BaseScenario.set_differentiation_method` of :class:`~gemseo.scenarios.base_scenario.BaseScenario`:

- for :code:`"finite_differences"` (default value):

.. code::

    scenario.set_differentiation_method("finite_differences")

- for the :code:`"complex_step"` method (each discipline must handle complex numbers):

.. code::

    scenario.set_differentiation_method("complex_step")

- for linearized version of the disciplines (:code:`"user"`): switching from direct mode to reverse mode is automatic, depending on the number of inputs and outputs. It can also be set by the user, setting :attr:`~gemseo.core.discipline.Discipline.linearization_mode` at :code:`"direct"` or :code:`"adjoint"`).

.. code::

    scenario.set_differentiation_method("user")
    for discipline in scenario.disciplines:
       discipline.linearization_mode='auto' # default, can also be 'direct' or 'adjoint'


When deriving a source code, it is very easy to make some errors or to forget to derive some terms: that is why implementation of derivation can be validated
against finite differences or complex step method, by means of the method :meth:`~gemseo.core.discipline.Discipline.check_jacobian`:

.. code::

    from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
    from gemseo.problems.mdo.sobieski.core import SobieskiProblem

    problem = SobieskiProblem("complex128")
    sr = SobieskiMission("complex128")
    sr.check_jacobian(indata, threshold=1e-12)

In order to be relevant, :code:`threshold` value should be kept at a low level
(typically :math:`<10^{-6}`).
