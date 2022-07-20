..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

Basics of MDO
*************
- Analyse your MDO problem and generate an N2 and XDSM diagram without wrapping any tool or writing code :ref:`MDO study analysis <gemseo_study>`
- Use different optimization algorithms  :ref:`See more... <optimization>`
- Use different sampling methods for :term:`design of experiments <DOE>`  :ref:`See more... <optimization>`
- Use different :ref:`MDO formulations <mdo_formulations>`: :ref:`MDF <mdf_formulation>`, :ref:`IDF <idf_formulation>`, :ref:`bilevel <bilevel_formulation>` and disciplinary optimizer  :ref:`See more... <mdo_formulations>`
- Visualize a :ref:`MDO formulation <mdo_formulations>` as an :ref:`XDSM diagram <xdsm>`  :ref:`See more... <mdo_formulations>`
- Use different :ref:`mda` algorithms: fixed-point algorithms (Gauss-Seidel and Jacobi), root finding methods (Newton Raphson and Quasi-Newton) and hybrid techniques  :ref:`See more... <mda>`
- Use different surrogate models to substitute a costly discipline within a process: linear regression, RBF model and Gaussian process regression  :ref:`See more... <surrogates>`
- Visualize the optimization results by means of many graphs  :ref:`See more... <post_processing>`
- Record and cache the disciplines inputs and outputs into :term:`HDF` files  :ref:`See more... <caching>`
- Experiment with different **|g|** 's benchmark :term:`MDO` problems  :ref:`See more... <benchmark_problems>`

Advanced techniques in MDO
**************************

- Create simple analytic disciplines using symbolic calculation  :ref:`See more... <analyticdiscipline>`
- Use a cheap scalable model instead of an costly discipline in order to compare different formulation performances  :ref:`See more... <scalable>`
- Monitor the execution of a scenario using logs, :ref:`XDSM <xdsm>` diagram or an observer design pattern :ref:`See more... <monitoring>`

Development
***********

- Interface simulation software with **|g|** using :term:`JSON` schema based grammars for inputs and output description and a wrapping class for execution  :ref:`See more... <software_connection>`

Plug-in options
***************

- Options of the available optimization algorithms  :ref:`See more... <gen_opt_algos>`
- Options of the available DOE algorithms  :ref:`See more... <gen_doe_algos>`
- Options of the available MDA algorithms  :ref:`See more... <gen_mda_algos>`
- Options of the available formulation algorithms  :ref:`See more... <gen_formulation_algos>`
- Options of the available post-processing algorithms  :ref:`See more... <gen_post_algos>`
- Options of the available machine learning algorithms  :ref:`See more... <gen_mlearning_algos>`
