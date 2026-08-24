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
from typing import Any
from typing import Final

from gemseo.dataset.io_dataset import IODataset
from gemseo.machine_learning.clustering.core.base_clusterer import BaseClusterer
from gemseo.machine_learning.core.model.base_supervised import BaseMLSupervisedModel
from gemseo.machine_learning.regression.core.base_regressor import BaseRegressor
from gemseo.machine_learning.transformer.scaler.min_max_scaler import MinMaxScaler
from gemseo.util.constant import READ_ONLY_EMPTY_DICT
from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    from gemseo.dataset.dataset import Dataset
    from gemseo.machine_learning.classification.core.base_classifier import (
        BaseClassifier,
    )
    from gemseo.machine_learning.classification.model.knn_settings import (
        KNNClassifier_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.classification.model.random_forest_settings import (
        RandomForestClassifier_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.classification.model.svm_settings import (
        SVMClassifier_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.clustering.model.gaussian_mixture_settings import (
        GaussianMixture_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.clustering.model.kmeans_settings import (
        KMeans_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.core.model.base_ml_model import BaseMLModel
    from gemseo.machine_learning.core.model.base_ml_model import TransformerType
    from gemseo.machine_learning.regression.model.fce_settings import (
        FCERegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.gpr_settings import (
        GaussianProcessRegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.gradient_boosting_settings import (
        GradientBoostingRegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.linreg_settings import (
        LinearRegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.mlp_settings import (
        MLPRegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.moe_settings import (
        MOERegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.ot_gpr_settings import (
        OTGaussianProcessRegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.pce_settings import (
        PCERegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.polyreg_settings import (
        PolynomialRegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.random_forest_settings import (
        RandomForestRegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.rbf_settings import (
        RBFRegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.regressor_chain_settings import (
        RegressorChain_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.svm_settings import (
        SVMRegressor_Settings,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.tps_settings import (
        TPSRegressor_Settings,  # noqa: F401
    )

LOGGER = logging.getLogger(__name__)


def get_mlearning_models() -> list[str]:
    """Get available machine learning models.

    Returns:
        The available machine learning models.
    """
    from gemseo.machine_learning.core.model.factory import ML_MODEL_FACTORY

    return ML_MODEL_FACTORY.class_names


def get_regression_models() -> list[str]:
    """Get available regression models.

    Returns:
        The available regression models.
    """
    from gemseo.machine_learning.regression.model.factory import REGRESSOR_FACTORY

    return REGRESSOR_FACTORY.class_names


def get_classification_models() -> list[str]:
    """Get available classification models.

    Returns:
        The available classification models.
    """
    from gemseo.machine_learning.classification.model.factory import CLASSIFIER_FACTORY

    return CLASSIFIER_FACTORY.class_names


def get_clustering_models() -> list[str]:
    """Get available clustering models.

    Returns:
        The available clustering models.
    """
    from gemseo.machine_learning.clustering.model.factory import CLUSTERER_FACTORY

    return CLUSTERER_FACTORY.class_names


def create_mlearning_model(
    name: str,
    data: Dataset,
    transformer: TransformerType = READ_ONLY_EMPTY_DICT,
    **parameters: Any,
) -> BaseMLModel:
    """Create a machine learning model from a training dataset.

    Args:
        name: The name of the machine learning model.
        data: The training dataset.
        transformer: The strategies to transform the variables.
            Values are instances of
            [BaseTransformer][gemseo.machine_learning.transformer.core.base_transformer.BaseTransformer]
            while keys are names of either variables or groups of variables.
            If
            [DEFAULT_TRANSFORMER][gemseo.machine_learning.core.model.base_ml_model.BaseMLModel.DEFAULT_TRANSFORMER],
            do not transform the variables.
        parameters: The parameters of the machine learning model.

    Returns:
        A machine learning model.
    """
    from gemseo.machine_learning.core.model.factory import ML_MODEL_FACTORY

    cls = ML_MODEL_FACTORY.get_class(name)
    settings = cls.settings_class(transformer=transformer, **parameters)
    return ML_MODEL_FACTORY.create(name, data, settings=settings)


minmax_inputs = {IODataset.INPUT_GROUP: MinMaxScaler()}


def create_regression_model(
    name: str,
    data: IODataset,
    transformer: TransformerType = BaseRegressor.DEFAULT_TRANSFORMER,  # noqa: E501
    **parameters: Any,
) -> BaseRegressor:
    """Create a regression model from a training dataset.

    Args:
        name: The name of the regression model.
        data: The training dataset.
        transformer: The strategies to transform the variables.
            Values are instances of
            [BaseTransformer][gemseo.machine_learning.transformer.core.base_transformer.BaseTransformer]
            while keys are names of either variables or groups of variables.
            If
            [DEFAULT_TRANSFORMER][gemseo.machine_learning.core.model.base_ml_model.BaseMLModel.DEFAULT_TRANSFORMER],
            do not transform the variables.
        parameters: The parameters of the regression model.

    Returns:
        A regression model.
    """
    from gemseo.machine_learning.regression.model.factory import REGRESSOR_FACTORY

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

    cls = REGRESSOR_FACTORY.get_class(name)
    settings = cls.settings_class(transformer=transformer, **parameters)
    return REGRESSOR_FACTORY.create(name, data, settings=settings)


def create_classification_model(
    name: str,
    data: IODataset,
    transformer: TransformerType = BaseMLSupervisedModel.DEFAULT_TRANSFORMER,
    # noqa: E501
    **parameters: Any,
) -> BaseClassifier:
    """Create a classification model from a training dataset.

    Args:
        name: The name of the classification model.
        data: The training dataset.
        transformer: The strategies to transform the variables.
            Values are instances of
            [BaseTransformer][gemseo.machine_learning.transformer.core.base_transformer.BaseTransformer]
            while keys are names of either variables or groups of variables.
            If
            [DEFAULT_TRANSFORMER][gemseo.machine_learning.core.model.base_ml_model.BaseMLModel.DEFAULT_TRANSFORMER],
            do not transform the variables.
        parameters: The parameters of the classification model.

    Returns:
        A classification model.
    """
    from gemseo.machine_learning.classification.model.factory import CLASSIFIER_FACTORY

    cls = CLASSIFIER_FACTORY.get_class(name)
    settings = cls.settings_class(transformer=transformer, **parameters)
    return CLASSIFIER_FACTORY.create(name, data, settings=settings)


def create_clustering_model(
    name: str,
    data: Dataset,
    transformer: TransformerType = BaseClusterer.DEFAULT_TRANSFORMER,
    **parameters: Any,
) -> BaseClusterer:
    """Create a clustering model from a training dataset.

    Args:
        name: The name of the clustering model.
        data: The training dataset.
        transformer: The strategies to transform the variables.
            Values are instances of
            [BaseTransformer][gemseo.machine_learning.transformer.core.base_transformer.BaseTransformer]
            while keys are names of either variables or groups of variables.
            If
            [DEFAULT_TRANSFORMER][gemseo.machine_learning.core.model.base_ml_model.BaseMLModel.DEFAULT_TRANSFORMER],
            do not transform the variables.
        parameters: The parameters of the clustering model.

    Returns:
        A clustering model.
    """
    from gemseo.machine_learning.clustering.model.factory import CLUSTERER_FACTORY

    cls = CLUSTERER_FACTORY.get_class(name)
    settings = cls.settings_class(transformer=transformer, **parameters)
    return CLUSTERER_FACTORY.create(name, data, settings=settings)


def get_mlearning_options(
    model_name: str, output_json: bool = False, pretty_print: bool = True
) -> dict[str, str] | str:
    """Find the available options for a machine learning model.

    Args:
        model_name: The name of the machine learning model.
        output_json: Whether to apply JSON format for the schema.
        pretty_print: Whether to print the schema in a pretty table.

    Returns:
        The options schema of the machine learning model.
    """
    from gemseo.machine_learning.core.model.factory import ML_MODEL_FACTORY

    return _get_options(ML_MODEL_FACTORY, model_name, output_json, pretty_print)


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
    from gemseo.machine_learning.regression.model.factory import REGRESSOR_FACTORY

    return _get_options(REGRESSOR_FACTORY, model_name, output_json, pretty_print)


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
    from gemseo.machine_learning.classification.model.factory import CLASSIFIER_FACTORY

    return _get_options(CLASSIFIER_FACTORY, model_name, output_json, pretty_print)


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
    from gemseo.machine_learning.clustering.model.factory import CLUSTERER_FACTORY

    return _get_options(CLUSTERER_FACTORY, model_name, output_json, pretty_print)


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

    schema = factory.get_class(model_name).settings_class.model_json_schema()
    if pretty_print:
        _pretty_print_schema(schema)
    if output_json:
        return json.dumps(schema)
    return schema


# Exported name -> "module.path:Attr" (lazy-loaded on attribute access).
# The module path is relative to ``gemseo.machine_learning``.
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "FCERegressor_Settings": "regression.model.fce_settings:FCERegressor_Settings",
    "GaussianMixture_Settings": (
        "clustering.model.gaussian_mixture_settings:GaussianMixture_Settings"
    ),
    "GaussianProcessRegressor_Settings": (
        "regression.model.gpr_settings:GaussianProcessRegressor_Settings"
    ),
    "GradientBoostingRegressor_Settings": (
        "regression.model.gradient_boosting_settings:GradientBoostingRegressor_Settings"
    ),
    "KMeans_Settings": "clustering.model.kmeans_settings:KMeans_Settings",
    "KNNClassifier_Settings": (
        "classification.model.knn_settings:KNNClassifier_Settings"
    ),
    "LinearRegressor_Settings": (
        "regression.model.linreg_settings:LinearRegressor_Settings"
    ),
    "MLPRegressor_Settings": "regression.model.mlp_settings:MLPRegressor_Settings",
    "MOERegressor_Settings": "regression.model.moe_settings:MOERegressor_Settings",
    "OTGaussianProcessRegressor_Settings": (
        "regression.model.ot_gpr_settings:OTGaussianProcessRegressor_Settings"
    ),
    "PCERegressor_Settings": "regression.model.pce_settings:PCERegressor_Settings",
    "PolynomialRegressor_Settings": (
        "regression.model.polyreg_settings:PolynomialRegressor_Settings"
    ),
    "RBFRegressor_Settings": "regression.model.rbf_settings:RBFRegressor_Settings",
    "RandomForestClassifier_Settings": (
        "classification.model.random_forest_settings:RandomForestClassifier_Settings"
    ),
    "RandomForestRegressor_Settings": (
        "regression.model.random_forest_settings:RandomForestRegressor_Settings"
    ),
    "RegressorChain_Settings": (
        "regression.model.regressor_chain_settings:RegressorChain_Settings"
    ),
    "SVMClassifier_Settings": (
        "classification.model.svm_settings:SVMClassifier_Settings"
    ),
    "SVMRegressor_Settings": "regression.model.svm_settings:SVMRegressor_Settings",
    "TPSRegressor_Settings": "regression.model.tps_settings:TPSRegressor_Settings",
}

_EXTRA_ALL: Final[tuple[str, ...]] = (
    "create_classification_model",
    "create_clustering_model",
    "create_mlearning_model",
    "create_regression_model",
    "get_classification_models",
    "get_classification_options",
    "get_clustering_models",
    "get_clustering_options",
    "get_mlearning_models",
    "get_mlearning_options",
    "get_regression_models",
    "get_regression_options",
)

install_lazy_reexport(globals(), _NAME_TO_LOCATION, extra_all=_EXTRA_ALL)
