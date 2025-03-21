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

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.clustering.algos.base_clusterer import BaseClusterer
from gemseo.mlearning.core.algos.supervised import BaseMLSupervisedAlgo
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.classification.algos.base_classifier import BaseClassifier
    from gemseo.mlearning.core.algos.ml_algo import BaseMLAlgo
    from gemseo.mlearning.core.algos.ml_algo import TransformerType

LOGGER = logging.getLogger(__name__)


def get_mlearning_models() -> list[str]:
    """Get available machine learning algorithms.

    Returns:
        The available machine learning algorithms.
    """
    from gemseo.mlearning.core.algos.factory import MLAlgoFactory

    return MLAlgoFactory().class_names


def get_regression_models() -> list[str]:
    """Get available regression models.

    Returns:
        The available regression models.
    """
    from gemseo.mlearning.regression.algos.factory import RegressorFactory

    return RegressorFactory().class_names


def get_classification_models() -> list[str]:
    """Get available classification models.

    Returns:
        The available classification models.
    """
    from gemseo.mlearning.classification.algos.factory import ClassifierFactory

    return ClassifierFactory().class_names


def get_clustering_models() -> list[str]:
    """Get available clustering models.

    Returns:
        The available clustering models.
    """
    from gemseo.mlearning.clustering.algos.factory import ClustererFactory

    return ClustererFactory().class_names


def create_mlearning_model(
    name: str,
    data: Dataset,
    transformer: TransformerType = READ_ONLY_EMPTY_DICT,
    **parameters,
) -> BaseMLAlgo:
    """Create a machine learning algorithm from a training dataset.

    Args:
        name: The name of the machine learning algorithm.
        data: The training dataset.
        transformer: The strategies to transform the variables.
            Values are instances of :class:`.BaseTransformer`
            while keys are names of either variables or groups of variables.
            If :attr:`~.BaseMLAlgo.IDENTITY`, do not transform the variables.
        parameters: The parameters of the machine learning algorithm.

    Returns:
        A machine learning model.
    """
    from gemseo.mlearning.core.algos.factory import MLAlgoFactory

    factory = MLAlgoFactory()
    return factory.create(name, data=data, transformer=transformer, **parameters)


minmax_inputs = {IODataset.INPUT_GROUP: MinMaxScaler()}


def create_regression_model(
    name: str,
    data: IODataset,
    transformer: TransformerType = BaseRegressor.DEFAULT_TRANSFORMER,  # noqa: E501
    **parameters,
) -> BaseRegressor:
    """Create a regression model from a training dataset.

    Args:
        name: The name of the regression algorithm.
        data: The training dataset.
        transformer: The strategies to transform the variables.
            Values are instances of :class:`.BaseTransformer`
            while keys are names of either variables or groups of variables.
            If :attr:`~.BaseMLAlgo.IDENTITY`, do not transform the variables.
        parameters: The parameters of the regression model.

    Returns:
        A regression model.
    """
    from gemseo.mlearning.regression.algos.factory import RegressorFactory

    if (
        name == "PCERegressor"
        and isinstance(transformer, Mapping)
        and IODataset.INPUT_GROUP in transformer
    ):
        LOGGER.warning(
            "Remove input data transformation because "
            "PCERegressor does not support transformers."
        )
        transformer = dict(transformer)
        del transformer[IODataset.INPUT_GROUP]
    factory = RegressorFactory()
    return factory.create(name, data=data, transformer=transformer, **parameters)


def create_classification_model(
    name: str,
    data: IODataset,
    transformer: TransformerType = BaseMLSupervisedAlgo.DEFAULT_TRANSFORMER,  # noqa: E501
    **parameters,
) -> BaseClassifier:
    """Create a classification model from a training dataset.

    Args:
        name: The name of the classification algorithm.
        data: The training dataset.
        transformer: The strategies to transform the variables.
            Values are instances of :class:`.BaseTransformer`
            while keys are names of either variables or groups of variables.
            If :attr:`~.BaseMLAlgo.IDENTITY`, do not transform the variables.
        parameters: The parameters of the classification model.

    Returns:
        A classification model.
    """
    from gemseo.mlearning.classification.algos.factory import ClassifierFactory

    return ClassifierFactory().create(
        name, data=data, transformer=transformer, **parameters
    )


def create_clustering_model(
    name: str,
    data: Dataset,
    transformer: TransformerType = BaseClusterer.DEFAULT_TRANSFORMER,
    **parameters,
) -> BaseClusterer:
    """Create a clustering model from a training dataset.

    Args:
        name: The name of the clustering algorithm.
        data: The training dataset.
        transformer: The strategies to transform the variables.
            Values are instances of :class:`.BaseTransformer`
            while keys are names of either variables or groups of variables.
            If :attr:`~.BaseMLAlgo.IDENTITY`, do not transform the variables.
        parameters: The parameters of the clustering model.

    Returns:
        A clustering model.
    """
    from gemseo.mlearning.clustering.algos.factory import ClustererFactory

    return ClustererFactory().create(
        name, data=data, transformer=transformer, **parameters
    )


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
    """
    from gemseo.mlearning.core.algos.factory import MLAlgoFactory

    return _get_options(MLAlgoFactory(), model_name, output_json, pretty_print)


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
    """
    from gemseo.mlearning.regression.algos.factory import RegressorFactory

    return _get_options(RegressorFactory(), model_name, output_json, pretty_print)


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
    """
    from gemseo.mlearning.classification.algos.factory import ClassifierFactory

    return _get_options(ClassifierFactory(), model_name, output_json, pretty_print)


def get_clustering_options(
    model_name: str, output_json: bool = False, pretty_print: bool = True
) -> dict[str, str] | str:
    """Find the available options for a clustering model.

    Args:
        model_name: The name of the clustering model.
        output_json: Whether to apply JSON format for the schema.
        pretty_print: Print the schema in a pretty table.

    Returns:
        The options schema of the clustering model.
    """
    from gemseo.mlearning.clustering.algos.factory import ClustererFactory

    return _get_options(ClustererFactory(), model_name, output_json, pretty_print)


def _get_options(
    factory, model_name, output_json, pretty_print
) -> dict[str, str] | str:
    """Find the available options for a model.

    Args:
        factory: The factory of model.
        model_name: The name of the model.
        output_json: Whether to apply JSON format for the schema.
        pretty_print: Print the schema in a pretty table.

    Returns:
        The options schema of the model.
    """
    from gemseo import _pretty_print_schema

    schema = factory.get_class(model_name).Settings.model_json_schema()
    if pretty_print:
        _pretty_print_schema(schema)
    if output_json:
        return json.dumps(schema)
    return schema
