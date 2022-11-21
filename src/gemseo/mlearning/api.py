# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Machine learning API.

The machine learning API provides methods for creating new and loading existing machine
learning models. It also provides methods for listing available models and options.
"""
from __future__ import annotations

import logging
from pathlib import Path

from gemseo.api import _get_schema
from gemseo.core.dataset import Dataset
from gemseo.mlearning.classification.classification import MLClassificationAlgo
from gemseo.mlearning.cluster.cluster import MLClusteringAlgo
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.core.supervised import MLSupervisedAlgo
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler

LOGGER = logging.getLogger(__name__)

# pylint: disable=import-outside-toplevel


def get_mlearning_models() -> list[str]:
    """Get available machine learning algorithms.

    Returns:
        The available machine learning algorithms.

    See also
    --------
    import_mlearning_model
    create_mlearning_model
    get_mlearning_options
    import_mlearning_model
    """
    from gemseo.mlearning.core.factory import MLAlgoFactory

    return MLAlgoFactory().models


def get_regression_models() -> list[str]:
    """Get available regression models.

    Returns:
        The available regression models.

    See also
    --------
    create_regression_model
    get_regression_options
    import_regression_model
    """
    from gemseo.mlearning.regression.factory import RegressionModelFactory

    return RegressionModelFactory().models


def get_classification_models() -> list[str]:
    """Get available classification models.

    Returns:
        The available classification models.

    See also
    --------
    create_classification_model
    get_classification_options
    import_classification_model
    """
    from gemseo.mlearning.classification.factory import ClassificationModelFactory

    return ClassificationModelFactory().models


def get_clustering_models() -> list[str]:
    """Get available clustering models.

    Returns:
        The available clustering models.

    See also
    --------
    create_clustering_model
    get_clustering_options
    import_clustering_model
    """
    from gemseo.mlearning.cluster.factory import ClusteringModelFactory

    return ClusteringModelFactory().models


def create_mlearning_model(
    name: str,
    data: Dataset,
    transformer: TransformerType = MLAlgo.IDENTITY,
    **parameters,
) -> MLAlgo:
    """Create a machine learning algorithm from a learning dataset.

    Args:
        name: The name of the machine learning algorithm.
        data: The learning dataset.
        transformer: The strategies to transform the variables.
            Values are instances of :class:`.Transformer`
            while keys are names of either variables or groups of variables.
            If :attr:`~.MLAlgo.IDENTITY`, do not transform the variables.
        parameters: The parameters of the machine learning algorithm.

    Returns:
        A machine learning model.

    See also
    --------
    get_mlearning_models
    get_mlearning_options
    import_mlearning_model
    """
    from gemseo.mlearning.core.factory import MLAlgoFactory

    factory = MLAlgoFactory()
    return factory.create(name, data=data, transformer=transformer, **parameters)


minmax_inputs = {Dataset.INPUT_GROUP: MinMaxScaler()}


def create_regression_model(
    name: str,
    data: Dataset,
    transformer: TransformerType = MLRegressionAlgo.DEFAULT_TRANSFORMER,  # noqa: B950
    **parameters,
) -> MLRegressionAlgo:
    """Create a regression model from a learning dataset.

    Args:
        name: The name of the regression algorithm.
        data: The learning dataset.
        transformer: The strategies to transform the variables.
            Values are instances of :class:`.Transformer`
            while keys are names of either variables or groups of variables.
            If :attr:`~.MLAlgo.IDENTITY`, do not transform the variables.
        parameters: The parameters of the regression model.

    Returns:
        A regression model.

    See also
    --------
    get_regression_models
    get_regression_options
    import_regression_model
    """
    from gemseo.mlearning.regression.factory import RegressionModelFactory

    factory = RegressionModelFactory()
    if (
        name == "PCERegressor"
        and isinstance(transformer, dict)
        and Dataset.INPUT_GROUP in transformer
    ):
        LOGGER.warning(
            "Remove input data transformation because "
            "PCERegressor does not support transformers."
        )
        del transformer[Dataset.INPUT_GROUP]
    return factory.create(name, data=data, transformer=transformer, **parameters)


def create_classification_model(
    name: str,
    data: Dataset,
    transformer: TransformerType = MLSupervisedAlgo.DEFAULT_TRANSFORMER,  # noqa: B950
    **parameters,
) -> MLClassificationAlgo:
    """Create a classification model from a learning dataset.

    Args:
        name: The name of the classification algorithm.
        data: The learning dataset.
        transformer: The strategies to transform the variables.
            Values are instances of :class:`.Transformer`
            while keys are names of either variables or groups of variables.
            If :attr:`~.MLAlgo.IDENTITY`, do not transform the variables.
        parameters: The parameters of the classification model.

    Returns:
        A classification model.

    See also
    --------
    get_classification_models
    get_classification_options
    import_classification_model
    """
    from gemseo.mlearning.classification.factory import ClassificationModelFactory

    return ClassificationModelFactory().create(
        name, data=data, transformer=transformer, **parameters
    )


def create_clustering_model(
    name: str,
    data: Dataset,
    transformer: TransformerType = MLClusteringAlgo.DEFAULT_TRANSFORMER,
    **parameters,
) -> MLClusteringAlgo:
    """Create a clustering model from a learning dataset.

    Args:
        name: The name of the clustering algorithm.
        data: The learning dataset.
        transformer: The strategies to transform the variables.
            Values are instances of :class:`.Transformer`
            while keys are names of either variables or groups of variables.
            If :attr:`~.MLAlgo.IDENTITY`, do not transform the variables.
        parameters: The parameters of the clustering model.

    Returns:
        A clustering model.

    See also
    --------
    get_clustering_models
    get_clustering_options
    import_clustering_model
    """
    from gemseo.mlearning.cluster.factory import ClusteringModelFactory

    return ClusteringModelFactory().create(
        name, data=data, transformer=transformer, **parameters
    )


def import_mlearning_model(directory: str | Path) -> MLAlgo:
    """Import a machine learning algorithm from a directory.

    Args:
        directory: The path to the directory.

    Returns:
        A machine learning model.

    See also
    --------
    create_mlearning_model
    get_mlearning_models
    get_mlearning_options
    """
    from gemseo.mlearning.core.factory import MLAlgoFactory

    return MLAlgoFactory().load(directory)


def import_regression_model(directory: str | Path) -> MLRegressionAlgo:
    """Import a regression model from a directory.

    Args:
        directory: The path of the directory.

    Returns:
        A regression model.

    See also
    --------
    create_regression_model
    get_regression_models
    get_regression_options
    """
    from gemseo.mlearning.regression.factory import RegressionModelFactory

    return RegressionModelFactory().load(directory)


def import_classification_model(directory: str | Path) -> MLClassificationAlgo:
    """Import a classification model from a directory.

    Args:
        directory: The path to the directory.

    Returns:
        A classification model.

    See also
    --------
    create_classification_model
    get_classification_models
    get_classification_options
    """
    from gemseo.mlearning.classification.factory import ClassificationModelFactory

    return ClassificationModelFactory().load(directory)


def import_clustering_model(directory: str | Path) -> MLClusteringAlgo:
    """Import a clustering model from a directory.

    Args:
        directory: The path to the directory.

    Returns:
        A clustering model.

    See also
    --------
    create_clustering_model
    get_clustering_models
    get_clustering_options
    """
    from gemseo.mlearning.cluster.factory import ClusteringModelFactory

    return ClusteringModelFactory().load(directory)


def get_mlearning_options(
    model_name: str, output_json: bool = False, pretty_print: bool = True
) -> dict[str, str] | str:
    """Find the available options for a machine learning algorithm.

    Args:
        model_name: The name of the machine learning algorithm.
        output_json: Whether to apply JSON format for the schema.
        pretty_print: Whether to print the schema in a pretty table.

    Returns:
        The options schema of the machine learning algorithm.

    See also
    --------
    create_mlearning_model
    get_mlearning_models
    import_mlearning_model
    """
    from gemseo.mlearning.core.factory import MLAlgoFactory

    return _get_schema(
        MLAlgoFactory().factory.get_options_grammar(model_name),
        output_json,
        pretty_print,
    )


def get_regression_options(
    model_name: str, output_json: bool = False, pretty_print: bool = True
) -> dict[str, str] | str:
    """Find the available options for a regression model.

    Args:
        model_name: The name of the regression model.
        output_json: Whether to apply JSON format for the schema.
        pretty_print: Print the schema in a pretty table.

    Returns:
        The options schema of the regression model.

    See also
    --------
    create_regression_model
    get_regression_models
    import_regression_model
    """
    from gemseo.mlearning.regression.factory import RegressionModelFactory

    return _get_schema(
        RegressionModelFactory().factory.get_options_grammar(model_name),
        output_json,
        pretty_print,
    )


def get_classification_options(
    model_name: str, output_json: bool = False, pretty_print: bool = True
) -> dict[str, str] | str:
    """Find the available options for a classification model.

    Args:
        model_name: The name of the classification model.
        output_json: Whether to apply JSON format for the schema.
        pretty_print: Print the schema in a pretty table.

    Returns:
        The options schema of the classification model.

    See also
    --------
    create_classification_model
    get_classification_models
    import_classification_model
    """
    from gemseo.mlearning.classification.factory import ClassificationModelFactory

    return _get_schema(
        ClassificationModelFactory().factory.get_options_grammar(model_name),
        output_json,
        pretty_print,
    )


def get_clustering_options(
    model_name: str, output_json: bool = False, pretty_print: bool = True
) -> dict[str, str] | str:
    """Find the available options for clustering model.

    Args:
        model_name: The name of the clustering model.
        output_json: Whether to apply JSON format for the schema.
        pretty_print: Print the schema in a pretty table.

    Returns:
        The options schema of the clustering model.

    See also
    --------
    create_clustering_model
    get_clustering_models
    import_clustering_model
    """
    from gemseo.mlearning.cluster.factory import ClusteringModelFactory

    return _get_schema(
        ClusteringModelFactory().factory.get_options_grammar(model_name),
        output_json,
        pretty_print,
    )
