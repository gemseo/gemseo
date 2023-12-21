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
"""Machine learning functionalities.

This module proposes many high-level functions for creating and loading machine learning
models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.clustering.clustering import MLClusteringAlgo
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.core.supervised import MLSupervisedAlgo
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.classification.classification import MLClassificationAlgo
    from gemseo.mlearning.core.ml_algo import TransformerType

LOGGER = logging.getLogger(__name__)


def get_mlearning_models() -> list[str]:
    """Get available machine learning algorithms.

    Returns:
        The available machine learning algorithms.

    See Also:
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

    See Also:
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

    See Also:
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

    See Also:
        create_clustering_model
        get_clustering_options
        import_clustering_model
    """
    from gemseo.mlearning.clustering.factory import ClusteringModelFactory

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

    See Also:
        get_mlearning_models
        get_mlearning_options
        import_mlearning_model
    """
    from gemseo.mlearning.core.factory import MLAlgoFactory

    factory = MLAlgoFactory()
    return factory.create(name, data=data, transformer=transformer, **parameters)


minmax_inputs = {IODataset.INPUT_GROUP: MinMaxScaler()}


def create_regression_model(
    name: str,
    data: IODataset,
    transformer: TransformerType = MLRegressionAlgo.DEFAULT_TRANSFORMER,  # noqa: E501
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

    See Also:
        get_regression_models
        get_regression_options
        import_regression_model
    """
    from gemseo.mlearning.regression.factory import RegressionModelFactory

    factory = RegressionModelFactory()
    if (
        name == "PCERegressor"
        and isinstance(transformer, dict)
        and IODataset.INPUT_GROUP in transformer
    ):
        LOGGER.warning(
            "Remove input data transformation because "
            "PCERegressor does not support transformers."
        )
        del transformer[IODataset.INPUT_GROUP]
    return factory.create(name, data=data, transformer=transformer, **parameters)


def create_classification_model(
    name: str,
    data: IODataset,
    transformer: TransformerType = MLSupervisedAlgo.DEFAULT_TRANSFORMER,  # noqa: E501
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

    See Also:
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

    See Also:
        get_clustering_models
        get_clustering_options
        import_clustering_model
    """
    from gemseo.mlearning.clustering.factory import ClusteringModelFactory

    return ClusteringModelFactory().create(
        name, data=data, transformer=transformer, **parameters
    )


def import_mlearning_model(directory: str | Path) -> MLAlgo:
    """Import a machine learning algorithm from a directory.

    Args:
        directory: The path to the directory.

    Returns:
        A machine learning model.

    See Also:
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

    See Also:
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

    See Also:
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

    See Also:
        create_clustering_model
        get_clustering_models
        get_clustering_options
    """
    from gemseo.mlearning.clustering.factory import ClusteringModelFactory

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

    See Also:
        create_mlearning_model
        get_mlearning_models
        import_mlearning_model
    """
    from gemseo import _get_schema
    from gemseo.mlearning.core.factory import MLAlgoFactory

    return _get_schema(
        MLAlgoFactory().get_options_grammar(model_name),
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

    See Also:
        create_regression_model
        get_regression_models
        import_regression_model
    """
    from gemseo import _get_schema
    from gemseo.mlearning.regression.factory import RegressionModelFactory

    return _get_schema(
        RegressionModelFactory().get_options_grammar(model_name),
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

    See Also:
        create_classification_model
        get_classification_models
        import_classification_model
    """
    from gemseo import _get_schema
    from gemseo.mlearning.classification.factory import ClassificationModelFactory

    return _get_schema(
        ClassificationModelFactory().get_options_grammar(model_name),
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

    See Also:
        create_clustering_model
        get_clustering_models
        import_clustering_model
    """
    from gemseo import _get_schema
    from gemseo.mlearning.clustering.factory import ClusteringModelFactory

    return _get_schema(
        ClusteringModelFactory().get_options_grammar(model_name),
        output_json,
        pretty_print,
    )
