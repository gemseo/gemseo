..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

Algorithms
~~~~~~~~~~

The data used to illustrate the post-processing algorithms in the following examples are
from an MDO scenario on the Sobieski's SSBJ problem.
The scenario uses the MDF formulation and was solved using the SciPy SLSQP algorithm.
The data have been saved in an HDF5 file, which is passed to the high-level function
:func:`.execute_post` along with the appropriate settings model.

.. note::

   To get the specific settings of a given post-processing, one should use the
   appropriate settings model accessible in :mod:`gemseo.settings.post`.
   Further details on the available post-processings and their settings can be found in
   the following dedicated page: :ref:`gen_post_algos`.

Details on how to execute such a scenario can be found
:ref:`here <sphx_glr_examples_formulations_plot_sobieski_mdf_example.py>`.
