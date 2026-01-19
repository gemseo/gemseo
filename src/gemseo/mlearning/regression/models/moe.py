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

from gemseo.algos.design_space import DesignSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.classification.models.factory import ClassifierFactory
from gemseo.mlearning.classification.quality.f1_measure import F1Measure
from gemseo.mlearning.clustering.models.factory import ClustererFactory
from gemseo.mlearning.clustering.quality.silhouette_measure import SilhouetteMeasure
from gemseo.mlearning.core.models.ml_model import MLModelSettingsType
from gemseo.mlearning.core.models.supervised import SavedObjectType as _SavedObjectType
from gemseo.mlearning.core.selection import MLModelSelection
from gemseo.mlearning.data_formatters.moe_data_formatters import MOEDataFormatters
from gemseo.mlearning.regression.models.base_regressor import BaseRegressor
from gemseo.mlearning.regression.models.factory import RegressorFactory
from gemseo.mlearning.regression.models.moe_settings import MOE_Settings
from gemseo.mlearning.regression.quality.mse_measure import MSEMeasure
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.classification.models.base_classifier import BaseClassifier
    from gemseo.mlearning.clustering.models.base_clusterer import BaseClusterer
    from gemseo.mlearning.core.models.ml_model import DataType
    from gemseo.mlearning.core.quality.base_ml_model_quality import BaseMLModelQuality
    from gemseo.mlearning.core.quality.base_ml_model_quality import (
        OptionType as EvalOptionType,
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

    clusterer_name: str
    """The clusterer's class name."""

    classifier_name: str
    """The classifier's class name."""

    regressor_name: str
    """The regressor's class name."""

    clusterer_settings: MLModelSettingsType
    """The clusterer's settings."""

    classifier_settings: MLModelSettingsType
    """The classifier's settings."""

    regressor_settings: MLModelSettingsType
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

    clusterer: BaseClusterer
    """The clustering model."""

    classifier: BaseClassifier
    """The classification model."""

    regressors: list[BaseRegressor]
    """The regression models."""

    SHORT_NAME: ClassVar[str] = "MoE"

    LABELS: Final[str] = "labels"

    _LOCAL_INPUT: Final[str] = "input"
    _LOCAL_OUTPUT: Final[str] = "output"

    DataFormatters = MOEDataFormatters

    Settings: ClassVar[type[MOE_Settings]] = MOE_Settings

    def _post_init(self):
        super()._post_init()
        self.hard = self._settings.hard
        self.clusterer_name = "KMeans"
        self.classifier_name = "KNNClassifier"
        self.regressor_name = "LinearRegressor"
        self.clusterer_settings = {}
        self.classifier_settings = {}
        self.regressor_settings = {}

        self.clustering_quality = None
        self.classification_quality = None
        self.regression_quality = None

        self.set_clustering_measure(SilhouetteMeasure)
        self.set_classification_measure(F1Measure)
        self.set_regression_measure(MSEMeasure)

        self.clustering_candidates = []
        self.regression_candidates = []
        self.classification_candidates = []

        self.clusterer = None
        self.classifier = None
        self.regressors = None

    def set_clusterer(
        self,
        name: str,
        **settings: MLModelSettingsType | None,
    ) -> None:
        """Set the clusterer.

        Args:
            name: The clusterer's class name.
            **settings: The clusterer's settings.
        """
        self.clusterer_name = name
        self.clusterer_settings = settings

    def set_classifier(
        self,
        name: str,
        **settings: MLModelSettingsType | None,
    ) -> None:
        """Set the classifier.

        Args:
            name: The classifier's class name.
            **settings: The classifier's settings.
        """
        self.classifier_name = name
        self.classifier_settings = settings

    def set_regressor(
        self,
        name: str,
        **settings: MLModelSettingsType | None,
    ) -> None:
        """Set the regressor.

        Args:
            name: The regressor's class name.
            **settings: The regressor's settings.
        """
        self.regressor_name = name
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
        name: str,
        calibration_space: DesignSpace | None = None,
        calibration_algorithm: dict[str, str | int] = READ_ONLY_EMPTY_DICT,
        **options: list[MLModelSettingsType] | None,
    ) -> None:
        """Add a candidate for clustering.

        Args:
            name: The name of a clustering model.
            calibration_space: The space
                defining the calibration variables.
            calibration_algorithm: The name and options of the DOE or optimization
                algorithm, e.g. {"algo": "fullfact", "n_samples": 10}).
                If `None`, do not perform calibration.
            **options: Parameters for the clustering model candidate.
                Each parameter has to be enclosed within a list.
                The list may contain different values to try out for the given
                parameter, or only one.
        """
        self.clustering_candidates.append(
            dict(
                name=name,
                calibration_space=calibration_space,
                calibration_algorithm=calibration_algorithm,
                **options,
            )
        )

    def add_classifier_candidate(
        self,
        name: str,
        calibration_space: DesignSpace | None = None,
        calibration_algorithm: dict[str, str | int] = READ_ONLY_EMPTY_DICT,
        **options: list[MLModelSettingsType] | None,
    ) -> None:
        """Add a candidate for classification.

        Args:
            name: The name of a classification model.
            calibration_space: The space
                defining the calibration variables.
            calibration_algorithm: The name and options of the DOE or optimization
                algorithm, e.g. {"algo": "fullfact", "n_samples": 10}).
                If `None`, do not perform calibration.
            **options: Parameters for the clustering model candidate.
                Each parameter has to be enclosed within a list.
                The list may contain different values to try out for the given
                parameter, or only one.
        """
        self.classification_candidates.append(
            dict(
                name=name,
                calibration_space=calibration_space,
                calibration_algorithm=calibration_algorithm,
                **options,
            )
        )

    def add_regressor_candidate(
        self,
        name: str,
        calibration_space: DesignSpace | None = None,
        calibration_algorithm: dict[str, str | int] = READ_ONLY_EMPTY_DICT,
        **options: list[MLModelSettingsType] | None,
    ) -> None:
        """Add a candidate for regression.

        Args:
            name: The name of a regression model.
            calibration_space: The space
                defining the calibration variables.
            calibration_algorithm: The name and options of the DOE or optimization
                algorithm, e.g. {"algo": "fullfact", "n_samples": 10}).
                If `None`, do not perform calibration.
            **options: Parameters for the regression model candidate.
                Each parameter has to be enclosed within a list.
                The list may contain different values to try out for the given
                parameter, or only one.
        """
        self.regression_candidates.append(
            dict(
                name=name,
                calibration_space=calibration_space,
                calibration_algorithm=calibration_algorithm,
                **options,
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
            factory = ClustererFactory()
            self.clusterer = factory.create(
                self.clusterer_name, data=dataset, **self.clusterer_settings
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
            factory = ClassifierFactory()
            self.classifier = factory.create(
                self.classifier_name,
                data=dataset,
                output_names=[self.LABELS],
                **self.classifier_settings,
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
        factory = RegressorFactory()
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
                local_model = factory.create(
                    self.regressor_name, data=dataset, **self.regressor_settings
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
