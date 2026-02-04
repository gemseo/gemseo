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
#                         documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Mixture of experts for regression.

The mixture of experts (MoE) model expresses an output variable
as the weighted sum of the outputs of local regression models,
whose weights depend on the input data.

During the learning stage,
the input space is divided into $K$ clusters by a clustering model,
then a classification model is built to map the input space to the cluster space,
and finally a regression model $f_k$ is built for the $k$-th cluster.

The classification may be either hard,
in which case only one of the weights is equal to one,
and the rest equal to zero:

$$y = \sum_{k=1}^K I_{C_k}(x) f_k(x),$$

or soft,
in which case the weights express the probabilities of belonging to each cluster:

$$y = \sum_{k=1}^K \mathbb{P}(x \in C_k) f_k(x),$$

where
$x$ is the input data,
$y$ is the output data,
$K$ is the number of clusters,
$(C_k)_{k=1,\cdots,K}$ are the input spaces associated to the clusters,
$I_{C_k}(x)$ is the indicator of class $k$,
$\mathbb{P}(x \in C_k)$ is the probability
that $x$ belongs to cluster $k$ and
$f_k(x)$ is the local regression model on cluster $k$.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import NoReturn

from numpy import newaxis
from numpy import nonzero
from numpy import unique
from numpy import zeros

from gemseo import REGRESSOR_FACTORY
from gemseo.algos.design_space import DesignSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.machine_learning.classification.models.factory import CLASSIFIER_FACTORY
from gemseo.machine_learning.classification.models.knn_settings import (
    KNNClassifier_Settings,
)
from gemseo.machine_learning.classification.quality.f1_measure import F1Measure
from gemseo.machine_learning.clustering.models.factory import CLUSTERER_FACTORY
from gemseo.machine_learning.clustering.models.kmeans_settings import KMeans_Settings
from gemseo.machine_learning.clustering.quality.silhouette_measure import (
    SilhouetteMeasure,
)
from gemseo.machine_learning.core.models.ml_model import MLModelSettingsType
from gemseo.machine_learning.core.models.supervised import (
    SavedObjectType as _SavedObjectType,
)
from gemseo.machine_learning.core.selection import MLModelSelection
from gemseo.machine_learning.data_formatters.moe_data_formatters import (
    MOEDataFormatters,
)
from gemseo.machine_learning.regression.models.base_regressor import BaseRegressor
from gemseo.machine_learning.regression.models.linreg_settings import (
    LinearRegressor_Settings,
)
from gemseo.machine_learning.regression.models.moe_settings import MOERegressor_Settings
from gemseo.machine_learning.regression.quality.mse_measure import MSEMeasure
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray

    from gemseo.algos.base_driver_settings import BaseDriverSettings
    from gemseo.datasets.dataset import Dataset
    from gemseo.machine_learning.classification.models.base_classifier import (
        BaseClassifier,
    )
    from gemseo.machine_learning.classification.models.base_classifier_settings import (
        BaseClassifierSettings,
    )
    from gemseo.machine_learning.clustering.models.base_clusterer import BaseClusterer
    from gemseo.machine_learning.clustering.models.base_clusterer_settings import (
        BaseClustererSettings,
    )
    from gemseo.machine_learning.core.models.ml_model import DataType
    from gemseo.machine_learning.core.quality.base_ml_model_quality import (
        BaseMLModelQuality,
    )
    from gemseo.machine_learning.core.quality.base_ml_model_quality import (
        OptionType as EvalOptionType,
    )
    from gemseo.machine_learning.regression.models.base_regressor_settings import (
        BaseRegressorSettings,
    )
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)

SavedObjectType = _SavedObjectType | str | dict

MLModelType = dict[
    str,
    str | DesignSpace | dict[str, str | int] | list[MLModelSettingsType] | None,
]


class MOERegressor(BaseRegressor):
    """Mixture of experts for regression."""

    hard: bool
    """Whether clustering/classification should be hard or soft."""

    clusterer_settings: BaseClustererSettings
    """The clusterer's settings."""

    classifier_settings: BaseClassifierSettings
    """The classifier's settings."""

    regressor_settings: BaseRegressorSettings
    """The regressor's settings."""

    clustering_quality: dict[str, str | EvalOptionType]
    """The quality measure for the clustering models."""

    classification_quality: dict[str, str | EvalOptionType]
    """The quality measure for the classification models."""

    regression_quality: dict[str, str | EvalOptionType]
    """The quality measure for the regression models."""

    clustering_candidates: list[MLModelType]
    """The clustering model candidates."""

    classification_candidates: list[MLModelType]
    """The classification model candidates."""

    regression_candidates: list[MLModelType]
    """The regression model candidates."""

    clusterer: BaseClusterer | None
    """The clustering model."""

    classifier: BaseClassifier | None
    """The classification model."""

    regressors: list[BaseRegressor]
    """The regression models."""

    SHORT_NAME: ClassVar[str] = "MoE"

    LABELS: Final[str] = "labels"

    _LOCAL_INPUT: Final[str] = "input"
    _LOCAL_OUTPUT: Final[str] = "output"

    DataFormatters = MOEDataFormatters

    settings_class: ClassVar[type[MOERegressor_Settings]] = MOERegressor_Settings

    def _post_init(self):
        super()._post_init()
        self.hard = self._settings.hard
        self.clusterer_settings = KMeans_Settings()
        self.classifier_settings = KNNClassifier_Settings()
        self.regressor_settings = LinearRegressor_Settings()

        self.clustering_quality = {}
        self.classification_quality = {}
        self.regression_quality = {}

        self.set_clustering_measure(SilhouetteMeasure)
        self.set_classification_measure(F1Measure)
        self.set_regression_measure(MSEMeasure)

        self.clustering_candidates = []
        self.regression_candidates = []
        self.classification_candidates = []

        self.clusterer = None
        self.classifier = None
        self.regressors = []

    def set_clusterer(self, settings: BaseClustererSettings) -> None:
        """Set the clusterer.

        Args:
            settings: The settings of the clusterer.
        """
        self.clusterer_settings = settings

    def set_classifier(self, settings: BaseClassifierSettings) -> None:
        """Set the classifier.

        Args:
            settings: The settings of the classifier.
        """
        self.classifier_settings = settings

    def set_regressor(self, settings: BaseRegressorSettings) -> None:
        """Set the regressor.

        Args:
            settings: The settings of the regressor.
        """
        self.regressor_settings = settings

    def set_clustering_measure(
        self,
        measure: BaseMLModelQuality,
        **options: EvalOptionType,
    ) -> None:
        """Set the quality measure for the clustering models.

        Args:
            measure: The quality measure.
            **options: The options for the quality measure.
        """
        self.clustering_quality = {
            "measure": measure,
            "options": options,
        }

    def set_classification_measure(
        self,
        measure: BaseMLModelQuality,
        **options: EvalOptionType,
    ) -> None:
        """Set the quality measure for the classification models.

        Args:
            measure: The quality measure.
            **options: The options for the quality measure.
        """
        self.classification_quality = {
            "measure": measure,
            "options": options,
        }

    def set_regression_measure(
        self,
        measure: BaseMLModelQuality,
        **options: EvalOptionType,
    ) -> None:
        """Set the quality measure for the regression models.

        Args:
            measure: The quality measure.
            **options: The options for the quality measure.
        """
        self.regression_quality = {
            "measure": measure,
            "options": options,
        }

    def add_clusterer_candidate(
        self,
        settings: BaseClustererSettings,
        calibration_space: DesignSpace | None = None,
        calibration_settings: BaseDriverSettings | None = None,
        **settings_catalogs: Iterable[MLModelSettingsType],
    ) -> None:
        """Add a clusterer candidate.

        Args:
            settings: The settings of the clusterer candidate.
            calibration_space: The space defining the settings to calibrate, if any.
            calibration_settings: The settings of the driver for calibration.
            **settings_catalogs: The catalogs of settings.
                Unlike the settings to calibrate,
                these settings are optimized using a grid search over the catalogs.
        """
        self.clustering_candidates.append(
            dict(
                settings=settings,
                calibration_space=calibration_space,
                calibration_settings=calibration_settings,
                **settings_catalogs,
            )
        )

    def add_classifier_candidate(
        self,
        settings: BaseClassifierSettings,
        calibration_space: DesignSpace | None = None,
        calibration_settings: BaseDriverSettings | None = None,
        **settings_catalogs: Iterable[MLModelSettingsType],
    ) -> None:
        """Add a classifier candidate.

        Args:
            settings: The settings of the classifier candidate.
            calibration_space: The space defining the settings to calibrate, if any.
            calibration_settings: The settings of the driver for calibration.
            **settings_catalogs: The catalogs of settings.
                Unlike the settings to calibrate,
                these settings are optimized using a grid search over the catalogs.
        """
        self.classification_candidates.append(
            dict(
                settings=settings,
                calibration_space=calibration_space,
                calibration_settings=calibration_settings,
                **settings_catalogs,
            )
        )

    def add_regressor_candidate(
        self,
        settings: BaseRegressorSettings,
        calibration_space: DesignSpace | None = None,
        calibration_settings: BaseDriverSettings | None = None,
        **settings_catalogs: Iterable[MLModelSettingsType],
    ) -> None:
        """Add a regressor candidate.

        Args:
            settings: The settings of the regressor candidate.
            calibration_space: The space defining the settings to calibrate, if any.
            calibration_settings: The settings of the driver for calibration.
            **settings_catalogs: The catalogs of settings.
                Unlike the settings to calibrate,
                these settings are optimized using a grid search over the catalogs.
        """
        self.regression_candidates.append(
            dict(
                settings=settings,
                calibration_space=calibration_space,
                calibration_settings=calibration_settings,
                **settings_catalogs,
            )
        )

    @DataFormatters.format_predict_class_dict
    @DataFormatters.format_samples()
    @DataFormatters.format_transform(transform_outputs=False)
    def predict_class(
        self,
        input_data: DataType,
    ) -> int | str | ndarray:
        """Predict classes from input data.

        The user can specify these input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{'a': array([1.]), 'b': array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.

        Returns:
            The predicted classes.
        """
        return self.classifier.predict(input_data)

    @DataFormatters.format_input_output()
    def predict_local_model(
        self,
        input_data: DataType,
        index: int,
    ) -> DataType:
        """Predict output data from input data.

        The user can specify these input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{'a': array([1.]), 'b': array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.
            index: The index of the local regression model.

        Returns:
            The predicted output data.
        """
        return self.regressors[index].predict(input_data)

    def _fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
    ) -> None:
        dataset = IODataset(dataset_name="training_set")
        dataset.add_group(
            dataset.INPUT_GROUP,
            input_data,
            [self._LOCAL_INPUT],
            {self._LOCAL_INPUT: input_data.shape[1]},
        )
        dataset.add_group(
            dataset.OUTPUT_GROUP,
            output_data,
            [self._LOCAL_OUTPUT],
            {self._LOCAL_OUTPUT: output_data.shape[1]},
        )
        self._fit_clusters(dataset)
        dataset_ = IODataset(dataset_name="training_set")
        dataset_.add_group(
            dataset.INPUT_GROUP,
            input_data,
            [self._LOCAL_INPUT],
            {self._LOCAL_INPUT: input_data.shape[1]},
        )
        dataset_.add_variable(
            self.LABELS, self.clusterer.labels[:, newaxis], dataset_.OUTPUT_GROUP
        )
        self._fit_classifier(dataset_)
        self._fit_regressors(dataset)

    def _fit_clusters(self, dataset: Dataset) -> None:
        """Train the clustering model.

        The method adds resulting labels as a new output in the dataset.

        Args:
            dataset: The dataset containing input and output data.
        """
        if not self.clustering_candidates:
            self.clusterer = CLUSTERER_FACTORY.create_from_settings(
                self.clusterer_settings, dataset
            )
            self.clusterer.learn()
        else:
            selector = MLModelSelection(
                dataset,
                self.clustering_quality["measure"],
                **self.clustering_quality["options"],
            )
            for cand in self.clustering_candidates:
                selector.add_candidate(**cand)
            self.clusterer = selector.select()
            LOGGER.info("Selected clusterer:")
            with MultiLineString.offset():
                LOGGER.info("%s", self.clusterer)

    def _fit_classifier(self, dataset: IODataset) -> None:
        """Train the classification model.

        Args:
            dataset: The dataset containing labeled input and output data.
        """
        if not self.classification_candidates:
            self.classifier_settings.output_names = [self.LABELS]
            self.classifier = CLASSIFIER_FACTORY.create_from_settings(
                self.classifier_settings, dataset
            )
            self.classifier.learn()
        else:
            selector = MLModelSelection(
                dataset,
                self.classification_quality["measure"],
                **self.classification_quality["options"],
            )
            for cand in self.classification_candidates:
                selector.add_candidate(output_names=[[self.LABELS]], **cand)
            self.classifier = selector.select()
            LOGGER.info("Selected classifier:")
            with MultiLineString.offset():
                LOGGER.info("%s", self.classifier)

    def _fit_regressors(self, dataset: IODataset) -> None:
        """Train the local regression models on each cluster separately.

        Args:
            dataset: The dataset containing labeled input and output data.
        """
        self.regressors = []
        for index in range(self.clusterer.n_clusters):
            samples = nonzero(self.clusterer.labels == index)[0].tolist()
            if self.regression_candidates:
                selector = MLModelSelection(
                    dataset,
                    self.regression_quality["measure"],
                    samples=samples,
                    **self.regression_quality["options"],
                )
                for cand in self.regression_candidates:
                    selector.add_candidate(**cand)
                local_model = selector.select()
                LOGGER.info("Selected regressor for cluster %s:", index)
                with MultiLineString.offset():
                    LOGGER.info("%s", local_model)
            else:
                local_model = REGRESSOR_FACTORY.create_from_settings(
                    self.regressor_settings, dataset
                )
                local_model.learn(samples=samples)

            self.regressors.append(local_model)

    def _predict_all(
        self,
        input_data: RealArray,
    ) -> RealArray:
        """Predict output of each regression model for given input data.

        This method stacks the different outputs along a new axis.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The output data with shape (n_samples, n_clusters, n_outputs).
        """
        # dim(input_data)  = (n_samples, n_inputs)
        # dim(output_data) = (n_samples, n_clusters, n_outputs)
        output_data = zeros((
            input_data.shape[0],
            self.n_clusters,
            self.regressors[0].output_dimension,
        ))
        for i in range(self.n_clusters):
            output_data[:, i] = self.regressors[i].predict(input_data)
        return output_data

    def _predict(
        self,
        input_data: RealArray,
    ) -> RealArray:
        # dim(probas)         = (n_samples, n_clusters,     1    )
        # dim(local_outputs)  = (n_samples, n_clusters, n_outputs)
        # dim(contributions)  = (n_samples, n_clusters, n_outputs)
        # dim(global_outputs) = (n_samples, n_outputs)
        probas = self.classifier.predict_proba(input_data, hard=self.hard)
        local_outputs = self._predict_all(input_data)
        contributions = probas * local_outputs
        return contributions.sum(axis=1)

    def _predict_jacobian(
        self,
        input_data: RealArray,
    ) -> RealArray:
        if self.hard:
            return self._predict_jacobian_hard(input_data)

        return self._predict_jacobian_soft(input_data)

    def _predict_jacobian_hard(
        self,
        input_data: RealArray,
    ) -> RealArray:
        """Predict the Jacobian matrices of the regression model at input_data.

        This method uses a hard classification.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The predicted Jacobian data with shape (n_samples, n_outputs, n_inputs).
        """
        classes = self.classifier.predict(input_data)[..., 0]
        first_regression_model = self.regressors[0]
        jacobians = zeros((
            len(input_data),
            first_regression_model.output_dimension,
            first_regression_model.input_dimension,
        ))
        for klass in unique(classes):
            inds_kls = (classes == klass).nonzero()[0]
            jacobians[inds_kls] = self.regressors[klass].predict_jacobian(
                input_data[inds_kls]
            )
        return jacobians

    def _predict_jacobian_soft(
        self,
        input_data: RealArray,
    ) -> NoReturn:
        """Predict the Jacobian matrices of the regression model at input_data.

        This method uses a soft classification.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The predicted Jacobian data with shape (n_samples, n_outputs, n_inputs).
        """
        raise NotImplementedError

    @property
    def __string_representation(self) -> MultiLineString:
        mls = super()._get_string_representation()
        mls.add("Clustering")
        mls.indent()
        mls.add(str(self.clusterer).split("\n")[0])
        mls.dedent()
        mls.add("Classification")
        mls.indent()
        mls.add(str(self.classifier).split("\n")[0])
        mls.dedent()
        mls.add("Regression")
        mls.indent()
        for i, local_model in enumerate(self.regressors):
            mls.add("Local model {}", i)
            mls.indent()
            mls.add(str(local_model).split("\n")[0])
            mls.dedent()
        return mls

    def __repr__(self) -> str:
        return str(self.__string_representation)

    def _repr_html_(self) -> str:
        return self.__string_representation._repr_html_()

    @property
    def labels(self) -> list[int]:
        """The cluster labels."""
        return self.clusterer.labels

    @property
    def n_clusters(self) -> int:
        """The number of clusters."""
        return self.clusterer.n_clusters
