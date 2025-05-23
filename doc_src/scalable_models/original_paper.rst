..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Charlie Vanaret, Francois Gallard

.. _scalable:

The scalable problem
====================

Introduction
------------

In this section we describe the |g|' scalable problem, or scalable discipline feature, based on the paper :cite:`Vanaret2017`:

.. code::

   @conference {VGM2017,
      title = {On the Consequences of the "No Free Lunch" Theorem for Optimization on the Choice of {MDO} Architecture},
      booktitle = {Proceedings of the AIAA SciTech Conference},
      year = {2017},
      month = {January},
      author = {Charlie Vanaret and Francois Gallard and Joaquim R. R. A. Martins}
   }

.. seealso::

   Scalable model is implemented in the class :class:`.DataDrivenScalableDiscipline` which inherited from :class:`.Discipline`.

.. seealso::

   The scalable model is illustrated in several examples: :ref:`sphx_glr_examples_scalable_plot_diagonal.py`, :ref:`sphx_glr_examples_scalable_plot_problem.py` and ref:`sphx_glr_download_examples_scalable_scalable_study.py`.

Based on computationally cheap disciplines, the scalable problem allows to choose a :ref:`MDO formulation <mdo_formulations>`:

- for the problem from which derives the scalable problem or
- for a family of problems having:

   - a greater number of design and coupling variables and
   - common properties with the original problem.

According to the authors, this scalable problem *"preserve[s] the functional characteristics of
the original problem and they proved useful in performing a rapid benchmarking of* :ref:`MDO formulation <mdo_formulations>`". This
*"provides insights on the scalability of MDO architectures with respect to the
dimensions of the problem. This may be achieved without having to execute the* :term:`MDO` *processes with the
original models. Our methodology thus requires a limited number of evaluations of the original models that
is independent of the desired dimensions of the design and the coupling variables of the scalable problem."*

Methodology
-----------

The proposed methodology

1. builds a surrogate model :math:`\Phi^{(int)}` for each discipline :math:`\Phi` of the initial problem with a limited amount of evaluations :math:`T`,
2. extrapolates the surrogate model :math:`\Phi^{(ext)}` to an arbitrary dimension.

The methodology preserves the interface of the initial problem, that is the names of the inputs (design variables) and the
outputs (coupling and state variables). Any high-fidelity discipline of
the initial problem may therefore be replaced by a cheap scalable
component generated by the methodology. :ref:`Strong properties <properties_scalable>` are guaranteed by the methodology.

One-dimensional restriction
***************************

The original model :math:`\Phi:\mathbb{R}^n\rightarrow\mathbb{R}^m` is restricted to a one-dimensional function :math:`\Phi^{(1d)}:[0,1]\rightarrow\mathbb{R}^m` by evaluating it along a diagonal line in the domain :math:`[x_1,\overline{x_1}]\times\ldots[x_n,\overline{x_n}]`:

.. math::

    \Phi^{(1d)}(t)=\Phi^{(1d)}\left(x_1+t(\overline{x_1}-x_1),\ldots,x_n+t(\overline{x_n}-x_n)\right)

Interpolation
*************

For any component :math:`i\in\{1,\ldots,m\}` of :math:`\Phi^{(1d)}`, the direct image of :math:`T` a finite subset of :math:`[0,1]` with cardinality :math:`|T|`, is:

.. math::

   \Phi_i^{(1d)}(T) = \left\{\Phi^{(1d)}(t)|t\in T\right\}

mapping from :math:`[0,1]` to :math:`[m_i, M_i]` and where :math:`m_i` and :math:`M_i` are respectively the minimal and maximal values reached by :math:`\Phi^{(1d)}` over :math:`T`.

The scaled version of :math:`\Phi_i^{(1d)}(T)` is

.. math::

   \Phi_i^{(s1d)}(T) = \left\{\Phi^{(1d)}(t)|t\in T\right\}

mapping from :math:`[0,1]` to :math:`[0,1]`.

Then, each component :math:`i` of :math:`\Phi^{(1d)}(t)` is approximated by a polynomial interpolation :math:`\Phi_i^{(int)}` over the date :math:`\left(T,\Phi^{(s1d)}(T)\right)`.

Input-output dependency
***********************

Dependencies between inputs and outputs can be represented by a sparse dependency matrix :math:`S` where:

- each block row represents a function of the problem (constraint or coupling),
- each block column represents an input (design variable or coupling),
- a nonzero element represents the dependency of a particular component of a function with respect to a particular component of an input.

In practice, the dependencies between inputs and outputs are not precisely known. Consequently, the matrix :math:`S` is randomly computed by block by means of a density factor (the filling of a block is proportional to this density factor).

Furthermore, initially taken in :math:`\mathcal{M}_{n,m}(\mathbb{R})`,this matrix :math:`S` can be taken in :math:`\mathcal{M}_{n_x,n_y}(\mathbb{R})` where the number of inputs :math:`n_x` and the number of outputs :math:`n_y` of the scalable model is freely chosen by the user.

Extrapolation
*************

Once :math:`n_x` and :math:`n_y` are chosen, we build the function :math:`\Phi^{(ext)}:[0,1]^{n_x}\rightarrow[0,1]^{n_y}` extrapolates :math:`\Phi^{(int)}:[0,1]\rightarrow[0,1]^{m}` to :math:`n_y` dimensions:

.. math::

   \Phi_i^{(ext)}(x)=\frac{1}{|S_{i.}|}\sum_{j\in S_{i.}} \Phi_{k_i}^{(int)}(x_j)

where:

- :math:`S_{i.}` represents the nonzero elements of the :math:`i`-th row of the dependency matrix :math:`S`.
- :math:`k_i` is an uniform random variable over :math:`\left\{1,\ldots,m\right\}`.

.. _properties_scalable:

Properties
----------

- **Existence of a solution to the coupling problem**. An equilibrium between all disciplines exists for any value of the design variables :math:`x`,
- **Preservation of ratio**. When :math:`n_y` approaches :math:`+\infty`, the ratio of components of the original functions is preserved.
- **Existence of a minimum**. There exists a feasible solution to the scalable problem, for any dimension of inputs and outputs.
- **Existence of derivatives**. The scalable extrapolations are continuously differentiable with respect to their inputs.
- **Existence of bounds on the target coupling variables**. All inputs and outputs belong to :math:`[0; 1]`,
  which ensures that all optimization variables are bounded, in particular coupling variables in IDF.
