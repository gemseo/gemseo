..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _machine_learning_overview:

Introduction to machine learning
================================

Introduction
------------

When an :class:`.MDODiscipline` is costly to evaluate, it can be replaced by
a :class:`.SurrogateDiscipline` cheap to evaluate, e.g. linear model, Kriging,
RBF regressor, ...
This :class:`.SurrogateDiscipline` is built from a few evaluations of this
:class:`.MDODiscipline`. This learning phase commonly relies on a regression
model calibrated by machine learning techniques. This is the reason why
|g| provides a machine learning package which includes the
:class:`.MLRegressionAlgo` class implementing the concept of regression model.
In addition, this machine learning package has a much broader set of features
than regression: clustering, classification, dimension reduction, data scaling,
...

.. seealso::

   :ref:`surrogates`

.. automodule:: gemseo.mlearning.core.ml_algo
   :noindex:

Development
-----------

This diagram shows the hierarchy of all machine learning algorithms,
and where they interact with :class:`.Dataset`, :class:`.MLQualityMeasure`,
:class:`.Transformer` and :class:`.MLAlgoCalibration`.

.. include:: machine_learning.txt
