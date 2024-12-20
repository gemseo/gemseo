..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

Calibration and selection
~~~~~~~~~~~~~~~~~~~~~~~~~

During the training stage,
the parameters of a machine learning model are modified
so that this model learns the training data as well as possible.

This model also depends on hyperparameters that are fixed during training.
For example, the polynomial degree in the case of polynomial regression.
The :class:`.MLAlgoCalibration` class can be used
to tune these hyperparameters so as to improve this model.

Moreover,
even if this model has learned well,
it is possible that another has learned better.
The :class:`.MLAlgoSelection` class can be used
to select the best machine learning model from a collection.
