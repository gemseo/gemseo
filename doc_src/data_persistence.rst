..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

:parenttoc: True
.. _data_persistence:

Data persistence
----------------

Many processes generate data that can be relevant for post-processing
(surrogate modelling, data analysis, visualization, ...).
Recording these data is called *data persistence*
and allows access to it at any time.

|g| offers different tools for data persistence:

- the cache stores the evaluations of a :class:`.MDODiscipline`,
- the database stores the evaluations of the :class:`.MDOFunction` instances attached to an :class:`.OptimizationProblem`,
- the dataset is a generic structure facilitating the post-processing of the data.

.. include:: includes/big_toc_css.rst
.. include:: tune_toc.rst
.. toctree::
   :maxdepth: 2

   Cache <cache>
   database
   dataset
