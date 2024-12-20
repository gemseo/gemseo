..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _machine_learning_examples:

Machine learning
================

This section illustrates the features of the :mod:`gemseo.mlearning` package:

- how to create a machine learning model for classification, clustering or regression,
- how to transform the data,
- how to assess the quality of a model,
- how to tune a model.

This package is particularly useful for creating a surrogate model,
as a :class:`.SurrogateDiscipline` wraps a regression model built from it.
However,
for those who are not very comfortable with machine learning
and for those who are essentially interested in the surrogate modelling,
starting with :ref:`the examples <surrogate_discipline_examples>`
about the :class:`.SurrogateDiscipline` should be more relevant.
