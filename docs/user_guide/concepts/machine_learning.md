---
description: "GEMSEO's machine learning capabilities,
covering clustering, classification, and regression models,
along with quality assessment, data transformation techniques, and model selection and calibration methods."
tags: ['user_guide']
search:
  boost: 2
---
<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Machine learning

## Introduction

Machine learning (ML) is the art of building models from data.
In GEMSEO, these models are called *ML models*.
A typical application of ML is
the construction of surrogate models to replace costly disciplines,
as explained in [this page][surrogate-models-introduction].

The data consist of $n$ samples:

$$\mathcal{L}=\left(p^{(i)}\right)_{1 \leq i \leq n},$$

describing $d$ properties of interest $p=(p_1,\ldots,p_d)$,
which can sometimes be organized into groups
such as inputs, outputs, or categories.
They are given to an ML model at instantiation
in the form of a [Dataset][gemseo.datasets.dataset.Dataset].

The ML model is fitted to the data
using the [learn()][gemseo.machine_learning.core.models.ml_model.BaseMLModel.learn] method;
it is then said to be *trained*.
Its quality can be assessed using specific mesures.

??? abstract "API"

    - [BaseMLModel][gemseo.machine_learning.core.models.ml_model.BaseMLModel]
      is the base class for ML models.
    - [BaseMLModelQuality][gemseo.machine_learning.core.quality.base_ml_model_quality.BaseMLModelQuality]
      is the base class for quality measures.

## ML models

### Clustering

In the absence of groups,
the data can be analyzed through a study of commonalities,
leading to $m$ plausible clusters $C_1,\ldots,C_m$,
with $C_i \subset \mathcal{L}$ and $C_i \cap C_j = \emptyset$.
This is referred to as *clustering*,
a branch of *unsupervised learning*
dedicated to the detection of patterns in unlabeled data.
A clustering model is also called a *clusterer*.
Once trained,
each training sample is assigned a category in $\{1,\ldots,m\}$
and most clusterers $\hat{f}$ can
[predict()][gemseo.machine_learning.clustering.models.base_predictive_clusterer.BasePredictiveClusterer.predict]
the category $c$ corresponding to new property values $x$,
i.e. $c \leftarrow \hat{f}(x)$.

he available clusterers are:

| Description      | Based on     | Class                                                                                                                                                                                                    |
|------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| k-means          | scikit-learn | [KMeans][gemseo.machine_learning.clustering.models.kmeans.KMeans] ([settings][gemseo.machine_learning.clustering.models.kmeans_settings.KMeans_Settings])                                                |
| Gaussian mixture | scikit-learn | [GaussianMixture][gemseo.machine_learning.clustering.models.gaussian_mixture.GaussianMixture] ([settings][gemseo.machine_learning.clustering.models.gaussian_mixture_settings.GaussianMixture_Settings]) |

!!! how-to

    - [Create a clustering model][create-a-clustering-model]

??? abstract "API"

    - [BaseMLUnsupervisedModel][gemseo.machine_learning.core.models.unsupervised.BaseMLUnsupervisedModel]
      is the base class for unsupervised ML models.
    - [BaseClusterer][gemseo.machine_learning.clustering.models.base_clusterer.BaseClusterer]
      is the base class for clustering models.

### Classification

When data can be separated into $m>2$ categories,
the purpose is to model the relations
between some properties of interest $x=(x_1,\ldots,x_d)$ and these categories $c\in\{1,\ldots,m\}$:

$$\mathcal{L}=\left(x^{(i)},c^{(i)}\right)_{1 \leq i \leq n}.$$

This is referred to as *classification*,
a branch of *supervised learning*.
In GEMSEO,
a classification model is also called a *classifier*
and the data must be passed to it in the form of an [IODataset][gemseo.datasets.io_dataset.IODataset].
Once trained,
a classifier $\hat{f}$ can
[predict()][gemseo.machine_learning.classification.models.base_classifier.BaseClassifier.predict]
the category $c$ corresponding to new property values $x$,
i.e. $c \leftarrow \hat{f}(x)$.

The available classifiers are:

| Description                  | Based on     | Class                                                                                                                                                                                                                        |
|------------------------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| k-nearest neighbors (k-NN)   | scikit-learn | [KNNClassifier][gemseo.machine_learning.classification.models.knn.KNNClassifier] ([settings][gemseo.machine_learning.classification.models.knn_settings.KNNClassifier_Settings])                                             |
| Random forest                | scikit-learn | [PolynomialRegressor][gemseo.machine_learning.classification.models.random_forest.RandomForestClassifier] ([settings][gemseo.machine_learning.classification.models.random_forest_settings.RandomForestClassifier_Settings]) |
| Support vector machine (SVM) | scikit-learn | [PolynomialRegressor][gemseo.machine_learning.classification.models.svm.SVMClassifier] ([settings][gemseo.machine_learning.classification.models.svm_settings.SVMClassifier_Settings])                                       |

!!! how-to

    - [Create a classification model][create-a-classification-model]

??? abstract "API"

    - [BaseMLSupervisedModel][gemseo.machine_learning.core.models.supervised.BaseMLSupervisedModel]
      is the base class for supervised ML model.
    - [BaseClassifier][gemseo.machine_learning.classification.models.base_classifier.BaseClassifier]
      is the base class for classification models.

### Regression

When the distinction between $d$ inputs $x=(x_1,\ldots,x_d)$ and $p$ outputs $y=(y_1,\ldots,y_p)$ can be made among the
properties,
i.e.:

$$\mathcal{L}=\left(x^{(i)},y^{(i)}\right)_{1 \leq i \leq n},$$

another branch of supervised learning can be considered,
namely *regression*.
In GEMSEO,
a regression model is also called a *regressor*
and the data must be passed to it in the form of an [IODataset][gemseo.datasets.io_dataset.IODataset].
Once trained,
a regressor $\hat{f}$ can
[predict()][gemseo.machine_learning.regression.models.base_regressor.BaseRegressor.predict]
the outputs $y$ corresponding to new inputs values $x$,
i.e. $y \leftarrow \hat{f}(x)$.
Most of the regressors can also
[predict_jacobian()][gemseo.machine_learning.regression.models.base_regressor.BaseRegressor.predict_jacobian],
i.e. $y' \leftarrow \nabla\hat{f}(x)=(\frac{\partial \hat{f}(x)}{\partial x_1},\ldots,\frac{\partial \hat{f}(x)}{\partial x_d})$.

The available regressors are:

| Description                      | Based on                   | Class                                                                                                                                                                                                                                    |
|----------------------------------|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Linear                           | scikit-learn               | [LinearRegressor][gemseo.machine_learning.regression.models.linreg.LinearRegressor] ([settings][gemseo.machine_learning.regression.models.linreg_settings.LinearRegressor_Settings])                                                     |
| Polynomial                       | scikit-learn               | [PolynomialRegressor][gemseo.machine_learning.regression.models.polyreg.PolynomialRegressor] ([settings][gemseo.machine_learning.regression.models.polyreg_settings.PolynomialRegressor_Settings])                                       |
| Radial basis function (RBF)      | SciPy                      | [RBFRegressor][gemseo.machine_learning.regression.models.rbf.RBFRegressor] ([settings][gemseo.machine_learning.regression.models.rbf_settings.RBFRegressor_Settings])                                                                    |
| Thin plate spline (TPS)          | SciPy                      | [TPSRegressor][gemseo.machine_learning.regression.models.thin_plate_spline.TPSRegressor] ([settings][gemseo.machine_learning.regression.models.thin_plate_spline_settings.TPSRegressor_Settings])                                        |
| Gaussian process (GP)            | scikit-learn               | [GaussianProcessRegressor][gemseo.machine_learning.regression.models.gpr.GaussianProcessRegressor] ([settings][gemseo.machine_learning.regression.models.gpr_settings.GaussianProcessRegressor_Settings])                                |
| Gaussian process (GP)            | OpenTURNS                  | [OTGaussianProcessRegressor][gemseo.machine_learning.regression.models.ot_gpr.OTGaussianProcessRegressor] ([settings][gemseo.machine_learning.regression.models.ot_gpr_settings.OTGaussianProcessRegressor_Settings])                    |
| Functional chaos expansion (FCE) | OpenTURNS and scikit-learn | [FCERegressor][gemseo.machine_learning.regression.models.fce.FCERegressor] ([settings][gemseo.machine_learning.regression.models.fce_settings.FCERegressor_Settings])                                                                    |
| Polynomial chaos expansion (PCE) | OpenTURNS                  | [PCERegressor][gemseo.machine_learning.regression.models.pce.PCERegressor] ([settings][gemseo.machine_learning.regression.models.pce_settings.PCERegressor_Settings])                                                                    |
| Multi-layer perceptron (MLP)     | scikit-learn               | [MLPRegressor][gemseo.machine_learning.regression.models.mlp.MLPRegressor] ([settings][gemseo.machine_learning.regression.models.mlp_settings.MLPRegressor_Settings])                                                                    |
| Support vector machine (SVM)     | scikit-learn               | [SVMRegressor][gemseo.machine_learning.regression.models.svm.SVMRegressor] ([settings][gemseo.machine_learning.regression.models.svm_settings.SVMRegressor_Settings])                                                                    |
| Gradient boosting                | scikit-learn               | [GradientBoostingRegressor][gemseo.machine_learning.regression.models.gradient_boosting.GradientBoostingRegressor] ([settings][gemseo.machine_learning.regression.models.gradient_boosting_settings.GradientBoostingRegressor_Settings]) |
| Random forest                    | scikit-learn               | [RandomForestRegressor][gemseo.machine_learning.regression.models.random_forest.RandomForestRegressor] ([settings][gemseo.machine_learning.regression.models.random_forest_settings.RandomForestRegressor_Settings])                     |

In addition,
regressors can be chained using the
[RegressorChain][gemseo.machine_learning.regression.models.regressor_chain.RegressorChain] regressor
([settings][gemseo.machine_learning.regression.models.regressor_chain_settings.RegressorChain_Settings]).
The idea is that the first regressor models the output
and each subsequent regressor models the error of the previous one.

Lastly,
GEMSEO implements the mixture-of-experts (MOE) algorithm
through the [MOERegressor][gemseo.machine_learning.regression.models.moe.MOERegressor] class
([settings][gemseo.machine_learning.regression.models.moe_settings.MOERegressor_Settings]).
The idea is to
split the training dataset into clusters,
then identify the relationship between the input variables and the clusters
and train local regressors onto these clusters.
Once trained,
given an new input point,
the MOE can either
predict the cluster and use the corresponding local regressor to predict the output value,
or predict the output value according to the different local regressors and average the predictions.

??? abstract "API"

    [BaseRegressor][gemseo.machine_learning.regression.models.base_regressor.BaseRegressor]
    is the base class for regression models.

## Quality assessment

In unsupervised learning,
the quality of a clusterer can represent the robustness of clusters definition
while in supervised learning,
the quality of an ML model can be interpreted as an error,
whether it is a misclassification in the case of the classifier
or a difference in value in the case of the regressor.

The quality of an ML model can be assessed in different ways.
Firstly,
in relation to the training dataset $\mathcal{L}$,
to determine whether this model has learned these data well.
Secondly,
in relation to a test dataset $\mathcal{T}$ (also called validation dataset),
to determine whether this model can generalize what it has learned to other data.
Then,
the challenge is to avoid over-learning the learning data
and thus lose in generality.
This problem is called *overfitting*.
In order to build ML models
that are not overly dependent on training data,
it is important to optimize both the learning and generalization qualities.

Since providing test data can be costly,
it is possible to approximate the generalization quality
by resampling the training dataset $\mathcal{L}$,
using techniques such as cross-validation and bootstrap.

The available quality measures are:

- for regression:

    | Description                    | Class                                                                              |
    |--------------------------------|------------------------------------------------------------------------------------|
    | Maximum error (ME)             | [MEMeasure][gemseo.machine_learning.regression.quality.me_measure.MEMeasure]       |
    | Mean absolute error (MAE)      | [MAEMeasure][gemseo.machine_learning.regression.quality.mae_measure.MAEMeasure]    |
    | Mean squared error (MSE)       | [MSEMeasure][gemseo.machine_learning.regression.quality.mse_measure.MSEMeasure]    |
    | Root mean squared error (RMSE) | [RMSEMeasure][gemseo.machine_learning.regression.quality.rmse_measure.RMSEMeasure] |
    | R² score                       | [R2Measure][gemseo.machine_learning.regression.quality.r2_measure.R2Measure]       |

- for classification

    | Description         | Class                                                                            |
    |---------------------|----------------------------------------------------------------------------------|
    | F1 score error (ME) | [MEMeasure][gemseo.machine_learning.classification.quality.f1_measure.F1Measure] |

- for clustering

    | Description            | Class                                                                                                |
    |------------------------|------------------------------------------------------------------------------------------------------|
    | Silhouette coefficient | [SilhouetteMeasure][gemseo.machine_learning.clustering.quality.silhouette_measure.SilhouetteMeasure] |

!!! tutorial

    - [Assessing the quality of an ML model][assessing-the-quality-of-an-ml-model].

!!! how-to

    - [Get the resampling result][get-the-resampling-result]
    - [Change the number of cross-validation folds][change-the-number-of-cross-validation-folds]
    - [Disable sample shuffling prior to cross validation][disable-sample-shuffling-prior-to-cross-validation]
    - [Make cross-validation and bootstrap reproducible][make-cross-validation-and-bootstrap-reproducible]

??? abstract "API"

    - [BaseMLModelQuality][gemseo.machine_learning.core.quality.base_ml_model_quality]
      is the base class for quality measures dedicated to regressors.
    - [BaseClustererQuality][gemseo.machine_learning.clustering.quality.base_clusterer_quality.BaseClustererQuality]
      is the base class for quality measures dedicated to clusterers.
    - [BasePredictiveClustererQuality][gemseo.machine_learning.clustering.quality.base_predictive_clusterer_quality.BasePredictiveClustererQuality]
      is the base class for quality measures dedicated to predictive clusterers.
    - [BaseClassifierQuality][gemseo.machine_learning.classification.quality.base_classifier_quality.BaseClassifierQuality]
      is the base class for quality measures dedicated to classifiers.
    - [BaseRegressorQuality][gemseo.machine_learning.regression.quality.base_regressor_quality.BaseRegressorQuality]
      is the base class for quality measures dedicated to regressors.

    ```mermaid
    classDiagram
        class BaseMLModelQuality {
            model
            compute_cross_validation_measure()
            compute_leave_one_out_measure()
            compute_test_measure()
            compute_learning_measure()
            is_better()
            SMALLER_IS_BETTER
        }
        class BaseMLModel {
            sampling_results
        }
        BaseMLModelQuality *-- BaseMLModel
    ```

## Data transformation

The quality of an ML model can often be improved
by transforming the input data and/or the output data.

### Scaling

Scaling data around zero is important to avoid numerical issues
when fitting a machine learning model.
This is all the more true as
the variables have different ranges
or the fitting relies on numerical optimization techniques.

The available data scalers are

| Description                                         | Class                                                                                 |
|-----------------------------------------------------|---------------------------------------------------------------------------------------|
| Scaling data into $[0,1]$ using minimum and maximum | [MinMaxScaler][gemseo.machine_learning.transformers.scaler.min_max_scaler.MinMaxScaler]      |
| Making data zero-mean and unit-variance             | [StandardScaler][gemseo.machine_learning.transformers.scaler.standard_scaler.StandardScaler] |

!!! how-to

    - [Scale data before training an ML model][scale-data-before-training-an-ml-model]

### Dimension reduction

With a constant budget,
the quality of ML models often decreases
when the input or output dimension increases,
because the number of parameters relative to the number of samples continues to grow,
which can lead to overfitting.
Furthermore,
some learning algorithms do not scale well with the input dimension.
Training the ML models using lower-dimensional data may then be appropriate.

The available data dimension techniques are

| Description                        | Based on     | Class                                                                  |
|------------------------------------|--------------|------------------------------------------------------------------------|
| Principal component analysis (PCA) | scikit-learn | [PCA][gemseo.machine_learning.transformers.dimension_reduction.pca.PCA]       |
| Kernel PCA (kPCA)                  | scikit-learn | [KPCA][gemseo.machine_learning.transformers.dimension_reduction.kpca.KPCA]    |
| Partial least squares (PLS)        | scikit-learn | [PLS][gemseo.machine_learning.transformers.dimension_reduction.pls.PLS]       |
| Karhunen-Loève SVD (KLSVD)         | OpenTURNS    | [KLSVD][gemseo.machine_learning.transformers.dimension_reduction.klsvd.KLSVD] |

!!! how-to

    - [Reduce the data dimension before training an ML model][reduce-the-data-dimension-before-training-an-ml-model]

### Reshaping

There are also power transformers for making the data more normally distributed:

| Description                      | Based on       | Class                                                                        |
|----------------------------------|----------------|------------------------------------------------------------------------------|
| Box-Cox power transformation     | scikit-learn   | [BoxCox][gemseo.machine_learning.transformers.power.boxcox.BoxCox]                  |
| Yeo-Johnson power transformation | scikit-learn   | [YeoJohnson][gemseo.machine_learning.transformers.power.yeo_johnson.YeoJohnson] |

### Chaining

Data transformations can be chained using a [Pipeline][gemseo.machine_learning.transformers.pipeline.Pipeline].

!!! how-to

    - [Chain data transformation][chain-data-transformations]

??? abstract "API"

    - [BaseTransformer][gemseo.machine_learning.transformers.base_transformer.BaseTransformer]
      is the base class for data transformers.
    - [Scaler][gemseo.machine_learning.transformers.scaler.scaler.Scaler]
      is the base class for data scalers.
    - [BaseDimensionReduction][gemseo.machine_learning.transformers.dimension_reduction.base_dimension_reduction.BaseDimensionReduction]
      is the base class for data dimension reduction technique.
    - [Power][gemseo.machine_learning.transformers.power.power.Power]
      is the base class for power transformers.

## Selection and fine-tuning

A ML model often depends on hyperparameters
to be carefully tuned in order to maximize the generalization quality of the model.
For example,
a polynomial regressor depends on the polynomial degree.
The [MLModelCalibration][gemseo.machine_learning.core.calibration.MLModelCalibration] class addresses this calibration problem,
either using a grid search on the hyperparameters values or an optimization from a calibration space.
In addition,
several ML models can be trained at the same time
to keep only the best,
using the [MLModelSelection][gemseo.machine_learning.core.selection.MLModelSelection] class.

!!! how-to

    - [Select an ML model][select-an-ml-model]
    - [Calibrate an ML model][calibrate-an-ml-model]
