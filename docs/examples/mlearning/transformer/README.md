<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!-- markdownlint-disable-next-line MD041 -->
## Data transformation

Fitting a model from transformed data rather than raw data can facilitate the training and improve the quality of the machine learning model. Every machine learning model has a `transformer` argument to set the transformation policy (none by default). In the special case of regression models, the function [create_surrogate()][gemseo.create_surrogate] and the [SurrogateDiscipline][gemseo.disciplines.surrogate.SurrogateDiscipline] class use the [BaseRegressor.DEFAULT_TRANSFORMER][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor.DEFAULT_TRANSFORMER] by default, which is [MinMaxScaler][gemseo.mlearning.transformers.scaler.min_max_scaler.MinMaxScaler] for both inputs and outputs.
