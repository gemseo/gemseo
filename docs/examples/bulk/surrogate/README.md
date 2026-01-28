<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Surrogate discipline examples

This section illustrates how to create and use a [SurrogateDiscipline][gemseo.disciplines.surrogate.SurrogateDiscipline].

This [Discipline][gemseo.core.discipline.discipline.Discipline] implements the notion of surrogate model, mainly used to approximate an expensive discipline from samples.

A [SurrogateDiscipline][gemseo.disciplines.surrogate.SurrogateDiscipline] wraps a regression model built from the [gemseo.mlearning][gemseo.mlearning] package. For those who are interested in machine learning techniques, such as data transformation and model assessment, please refer to [the corresponding examples][machine-learning-examples].
