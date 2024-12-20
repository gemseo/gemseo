..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

Data transformation
~~~~~~~~~~~~~~~~~~~

Fitting a model from transformed data rather than raw data can facilitate the training
and improve the quality of the machine learning model.
Every machine learning model has a ``transformer`` argument to set the transformation policy (none by default).
In the special case of regression models,
the function :func:`.create_surrogate` and the :class:`.SurrogateDiscipline` class
use the :attr:`.BaseRegressor.DEFAULT_TRANSFORMER` by default,
which is :class:`.MinMaxScaler` for both inputs and outputs.
