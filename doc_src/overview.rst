..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

.. _overview:

Overview
========

|g| stands for Generic Engine for Multi-disciplinary Scenarios, Exploration and Optimization.

Built on top of `NumPy  <http://www.numpy.org/>`_, `SciPy  <http://www.scipy.org/>`_ and `Matplotlib  <http://www.matplotlib.org/>`_ libraries,
this `Python <https://www.python.org/>`__ library enables an automatic treatment of design problems,
using design of experiments, optimization algorithms and graphic post-processing.

|g| is more particularly specialized in `Multidisciplinary Design Optimization <https://en.wikipedia.org/wiki/Multidisciplinary_design_optimization>`_ (:term:`MDO`).

What is |g| used for?
---------------------

Let us consider a design problem to be solved
over a particular design space
and using different :term:`simulation software` (called :term:`disciplines <discipline>`).

From this information, |g| carries out the following steps:

   #. **Create** a :term:`scenario`, translating this design problem into a mathematical :term:`optimization problem`.
      The user can choose its favorite :ref:`MDO formulation <mdo_formulations>` (or architecture) or test another one,
      by simply giving the name of the formulation to the scenario.
      :ref:`Bi-level MDO formulations <bilevel_formulation>` allow to use disciplines that are themselves scenarios.

   #. **Solve** this design problem, using either an :term:`optimization algorithm` or a :term:`DOE`.

   #. **Plot** the results and logs of the optimization problem resolution.

The power of |g|
----------------

Fully based on the power of object-oriented programming,
the :term:`scenario` automatically generates the :term:`process`,
with corresponding :term:`work flow` and :term:`data flow`.

|g| aims at pushing forward the limits of automation in simulation processes development,
with a particular focus made on:

   - the integration of heterogeneous simulation environments in industrial and research contexts,
   - the integration of state of the art algorithms for optimization, design of experiments, and coupled analyses,
   - the automation of :term:`MDO` results analysis,
   - the development of distributed and multi-level :ref:`MDO formulations <mdo_formulations>`.

.. figure:: /_images/bilevel_ssbj.png

   A bi-level MDO formulation on Sobieski's SSBJ test case.

Main features
-------------

Basics of MDO
*************
- Analyse an MDO problem and generate an N2 chart and an XDSM diagram without wrapping any tool or writing code :ref:`[Read more] <gemseo_study>`
- Use different optimization algorithms  :ref:`[Read more] <optimization>`
- Use different sampling methods for :term:`design of experiments <DOE>`  :ref:`[Read more] <optimization>`
- Use different :ref:`MDO formulations <mdo_formulations>`: :ref:`MDF <mdf_formulation>`, :ref:`IDF <idf_formulation>`, :ref:`bilevel <bilevel_formulation>` and disciplinary optimizer  :ref:`[Read more] <mdo_formulations>`
- Visualize a :ref:`MDO formulation <mdo_formulations>` as an :ref:`XDSM diagram <xdsm>`  :ref:`[Read more] <mdo_formulations>`
- Use different :ref:`mda` algorithms: fixed-point algorithms (Gauss-Seidel and Jacobi), root finding methods (Newton Raphson and Quasi-Newton) and hybrid techniques  :ref:`[Read more] <mda>`
- Use different surrogate models to substitute a costly discipline within a process: linear regression, RBF model and Gaussian process regression  :ref:`[Read more] <surrogates>`
- Visualize the optimization results by means of many graphs  :ref:`[Read more] <post_processing>`
- Record and cache the disciplines inputs and outputs into :term:`HDF` files  :ref:`[Read more] <caching>`
- Experiment with different |g| 's benchmark :term:`MDO` problems  :ref:`[Read more] <benchmark_problems>`

Advanced techniques in MDO
**************************

- Create simple analytic disciplines using symbolic calculation  :ref:`[Read more] <analyticdiscipline>`
- Use a cheap scalable model instead of an costly discipline in order to compare different formulation performances  :ref:`[Read more] <scalable>`
- Monitor the execution of a scenario using logs, :ref:`XDSM <xdsm>` diagram or an observer design pattern :ref:`[Read more] <monitoring>`

Development
***********

- Interface simulation software with |g| using :term:`JSON` schema based grammars for inputs and output description and a wrapping class for execution  :ref:`[Read more] <software_connection>`

Plug-in options
***************

- Options of the available optimization algorithms  :ref:`[Read more] <gen_opt_algos>`
- Options of the available DOE algorithms  :ref:`[Read more] <gen_doe_algos>`
- Options of the available MDA algorithms  :ref:`[Read more] <gen_mda_algos>`
- Options of the available formulation algorithms  :ref:`[Read more] <gen_formulation_algos>`
- Options of the available post-processing algorithms  :ref:`[Read more] <gen_post_algos>`
- Options of the available machine learning algorithms  :ref:`[Read more] <gen_mlearning_algos>`
