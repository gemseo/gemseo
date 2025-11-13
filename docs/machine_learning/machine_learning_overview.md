<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Introduction to machine learning

## Introduction

When a [Discipline][gemseo.core.discipline.discipline.Discipline] is costly to evaluate,
it can be replaced by a [SurrogateDiscipline][gemseo.disciplines.surrogate.SurrogateDiscipline] cheap to evaluate,
e.g. linear model, Kriging, RBF regressor, ...
This [SurrogateDiscipline][gemseo.disciplines.surrogate.SurrogateDiscipline] is built from a few evaluations
of this [Discipline][gemseo.core.discipline.discipline.Discipline].
This learning phase commonly relies on a regression
model calibrated by machine learning techniques. This is the reason why
GEMSEO provides a machine learning package which includes the
[BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor] class implementing the concept of regression model.
In addition, this machine learning package has a much broader set of features
than regression: clustering, classification, dimension reduction, data scaling,
...

!!! info "See Also"

      [Surrogate models](../surrogate.md)

## Development

This diagram shows the hierarchy of all machine learning algorithms,
and where they interact with [Dataset][gemseo.datasets.dataset.Dataset],
[BaseMLAlgoQuality][gemseo.mlearning.core.quality.base_ml_algo_quality.BaseMLAlgoQuality],
[BaseTransformer][gemseo.mlearning.transformers.base_transformer.BaseTransformer]
and [MLAlgoCalibration][gemseo.mlearning.core.calibration.MLAlgoCalibration].

```mermaid
classDiagram
    class Dataset {
    }

    class BaseMLAlgo {
        <<abstract>>
       +SHORT_ALGO_NAME
       +LIBRARY
       +algo
       +is_trained
       +learning_set
       +parameters
       +transformer
       +DataFormatters
       +learn()
       +save()
       #save_algo()
       +load_algo()
       #get_objects_to_save()
    }

    class BaseMLUnsupervisedAlgo {
        <<abstract>>
       +var_names
       +learn()
       #fit()
    }

    class BaseClusterer {
        <<abstract>>
       +n_clusters
       +labels
       +learn()
       +predict()
       +predict_proba()
       #predict_proba()
       #predict_proba_hard()
       #predict_proba_soft()
    }

    class BaseMLSupervisedAlgo {
        <<abstract>>
       +input_names
       +input_dimension
       +output_names
       +output_dimension
       +DataFormatters
       +learn()
       +predict()
       #fit()
       #predict()
       #get_objects_to_save()
    }

    class BaseClassifier {
        <<abstract>>
       +n_classes
       +learn()
       +predict_proba()
       #predict_proba()
       #predict_proba_hard()
       #predict_proba_soft()
       #get_objects_to_save()
    }

    class BaseRegressor {
        <<abstract>>
       +DataFormatters
       +predict_raw()
       +predict_jacobian()
       #predict_jacobian()
    }

    class BaseMLAlgoQuality {
        <<abstract>>
    }

    class SurrogateDiscipline {
        <<abstract>>
       +input_grammar
       +output_grammar
       +execute()
       +linearize()
    }

    class BaseTransformer {
        <<abstract>>
       +duplicate()
       +fit()
       +transform()
       +inverse_transform()
       +fit_transform()
       +compute_jacobian()
       +compute_jacobian_inverse()
    }


    BaseMLAlgo *-- Dataset
    BaseMLAlgo *-- BaseTransformer
    BaseMLAlgo <|-- BaseMLUnsupervisedAlgo
    BaseMLAlgo <|-- BaseMLSupervisedAlgo
    BaseMLUnsupervisedAlgo <|-- BaseClusterer
    BaseMLSupervisedAlgo <|-- BaseRegressor
    BaseMLSupervisedAlgo <|-- BaseClassifier
    BaseMLAlgoQuality *-- BaseMLAlgo
    SurrogateDiscipline *-- BaseRegressor
```
