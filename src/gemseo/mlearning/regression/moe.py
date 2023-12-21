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
the input space is divided into :math:`K` clusters by a clustering model,
then a classification model is built to map the input space to the cluster space,
and finally a regression model :math:`f_k` is built for the :math:`k`-th cluster.

The classification may be either hard,
in which case only one of the weights is equal to one,
and the rest equal to zero:

.. math::

    y = \sum_{k=1}^K I_{C_k}(x) f_k(x),

or soft,
in which case the weights express the probabilities of belonging to each cluster:

.. math::

    y = \sum_{k=1}^K \mathbb{P}(x \in C_k) f_k(x),

where
:math:`x` is the input data,
:math:`y` is the output data,
:math:`K` is the number of clusters,
:math:`(C_k)_{k=1,\cdots,K}` are the input spaces associated to the clusters,
:math:`I_{C_k}(x)` is the indicator of class :math:`k`,
:math:`\mathbb{P}(x \in C_k)` is the probability
that :math:`x` belongs to cluster :math:`k` and
:math:`f_k(x)` is the local regression model on cluster :math:`k`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import NoReturn
from typing import Optional
from typing import Union

from numpy import ndarray
from numpy import newaxis
from numpy import nonzero
from numpy import unique
from numpy import zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.classification.factory import ClassificationModelFactory
from gemseo.mlearning.clustering.factory import ClusteringModelFactory
from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.core.selection import MLAlgoSelection
from gemseo.mlearning.core.supervised import SavedObjectType as _SavedObjectType
from gemseo.mlearning.data_formatters.moe_data_formatters import MOEDataFormatters
from gemseo.mlearning.quality_measures.f1_measure import F1Measure
from gemseo.mlearning.quality_measures.mse_measure import MSEMeasure
from gemseo.mlearning.quality_measures.silhouette_measure import SilhouetteMeasure
from gemseo.mlearning.regression.factory import RegressionModelFactory
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.classification.classification import MLClassificationAlgo
    from gemseo.mlearning.clustering.clustering import MLClusteringAlgo
    from gemseo.mlearning.quality_measures.quality_measure import MLQualityMeasure
    from gemseo.mlearning.quality_measures.quality_measure import (
        OptionType as EvalOptionType,
    )

LOGGER = logging.getLogger(__name__)

SavedObjectType = Union[_SavedObjectType, str, dict]

MLAlgoType = dict[
    str,
    Optional[
        Union[str, DesignSpace, dict[str, Union[str, int]], list[MLAlgoParameterType]]
    ],
]


class MOERegressor(MLRegressionAlgo):
    """Mixture of experts regression."""

    hard: bool
    """Whether clustering/classification should be hard or soft."""

    cluster_algo: str
    """The name of the clustering algorithm."""

    classif_algo: str
    """The name of the classification algorithm."""

    regress_algo: str
    """The name of the regression algorithm."""

    cluster_params: MLAlgoParameterType
    """The parameters of the clustering algorithm."""

    classif_params: MLAlgoParameterType
    """The parameters of the classification algorithm."""

    regress_params: MLAlgoParameterType
    """The parameters of the regression algorithm."""

    cluster_measure: dict[str, str | EvalOptionType]
    """The quality measure for the clustering algorithms."""

    classif_measure: dict[str, str | EvalOptionType]
    """The quality measure for the classification algorithms."""

    regress_measure: dict[str, str | EvalOptionType]
    """The quality measure for the regression algorithms."""

    cluster_cands: list[MLAlgoType]
    """The clustering algorithm candidates."""

    classif_cands: list[MLAlgoType]
    """The classification algorithm candidates."""

    regress_cands: list[MLAlgoType]
    """The regression algorithm candidates."""

    clusterer: MLClusteringAlgo
    """The clustering algorithm."""

    classifier: MLClassificationAlgo
    """The classification algorithm."""

    regress_models: list[MLRegressionAlgo]
    """The regression algorithms."""

    SHORT_ALGO_NAME: ClassVar[str] = "MoE"

    LABELS: Final[str] = "labels"

    _LOCAL_INPUT: Final[str] = "input"
    _LOCAL_OUTPUT: Final[str] = "output"

    DataFormatters = MOEDataFormatters

    def __init__(
        self,
        data: IODataset,
        transformer: TransformerType = MLRegressionAlgo.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        hard: bool = True,
    ) -> None:
        """
        Args:
            hard: Whether clustering/classification should be hard or soft.
        """  # noqa: D205 D212
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            hard=hard,
        )
        self.hard = hard
        self.cluster_algo = "KMeans"
        self.classif_algo = "KNNClassifier"
        self.regress_algo = "LinearRegressor"
        self.cluster_params = {}
        self.classif_params = {}
        self.regress_params = {}

        self.cluster_measure = None
        self.classif_measure = None
        self.regress_measure = None

        self.set_clustering_measure(SilhouetteMeasure)
        self.set_classification_measure(F1Measure)
        self.set_regression_measure(MSEMeasure)

        self.cluster_cands = []
        self.regress_cands = []
        self.classif_cands = []

        self.clusterer = None
        self.classifier = None
        self.regress_models = None

    def set_clusterer(
        self,
        cluster_algo: str,
        **cluster_params: MLAlgoParameterType | None,
    ) -> None:
        """Set the clustering algorithm.

        Args:
            cluster_algo: The name of the clustering algorithm.
            **cluster_params: The parameters of the clustering algorithm.
        """
        self.cluster_algo = cluster_algo
        self.cluster_params = cluster_params

    def set_classifier(
        self,
        classif_algo: str,
        **classif_params: MLAlgoParameterType | None,
    ) -> None:
        """Set the classification algorithm.

        Args:
            classif_algo: The name of the classification algorithm.
            **classif_params: The parameters of the classification algorithm.
        """
        self.classif_algo = classif_algo
        self.classif_params = classif_params

    def set_regressor(
        self,
        regress_algo: str,
        **regress_params: MLAlgoParameterType | None,
    ) -> None:
        """Set the regression algorithm.

        Args:
            regress_algo: The name of the regression algorithm.
            **regress_params: The parameters of the regression algorithm.
        """
        self.regress_algo = regress_algo
        self.regress_params = regress_params

    def set_clustering_measure(
        self,
        measure: MLQualityMeasure,
        **eval_options: EvalOptionType,
    ) -> None:
        """Set the quality measure for the clustering algorithms.

        Args:
            measure: The quality measure.
            **eval_options: The options for the quality measure.
        """
        self.cluster_measure = {
            "measure": measure,
            "options": eval_options,
        }

    def set_classification_measure(
        self,
        measure: MLQualityMeasure,
        **eval_options: EvalOptionType,
    ) -> None:
        """Set the quality measure for the classification algorithms.

        Args:
            measure: The quality measure.
            **eval_options: The options for the quality measure.
        """
        self.classif_measure = {
            "measure": measure,
            "options": eval_options,
        }

    def set_regression_measure(
        self,
        measure: MLQualityMeasure,
        **eval_options: EvalOptionType,
    ) -> None:
        """Set the quality measure for the regression algorithms.

        Args:
            measure: The quality measure.
            **eval_options: The options for the quality measure.
        """
        self.regress_measure = {
            "measure": measure,
            "options": eval_options,
        }

    def add_clusterer_candidate(
        self,
        name: str,
        calib_space: DesignSpace | None = None,
        calib_algo: dict[str, str | int] | None = None,
        **option_lists: list[MLAlgoParameterType] | None,
    ) -> None:
        """Add a candidate for clustering.

        Args:
            name: The name of a clustering algorithm.
            calib_space: The space
                defining the calibration variables.
            calib_algo: The name and options of the DOE or optimization
                algorithm, e.g. {"algo": "fullfact", "n_samples": 10}).
                If ``None``, do not perform calibration.
            **option_lists: Parameters for the clustering algorithm candidate.
                Each parameter has to be enclosed within a list.
                The list may contain different values to try out for the given
                parameter, or only one.
        """
        self.cluster_cands.append(
            dict(
                name=name,
                calib_space=calib_space,
                calib_algo=calib_algo,
                **option_lists,
            )
        )

    def add_classifier_candidate(
        self,
        name: str,
        calib_space: DesignSpace | None = None,
        calib_algo: dict[str, str | int] | None = None,
        **option_lists: list[MLAlgoParameterType] | None,
    ) -> None:
        """Add a candidate for classification.

        Args:
            name: The name of a classification algorithm.
            calib_space: The space
                defining the calibration variables.
            calib_algo: The name and options of the DOE or optimization
                algorithm, e.g. {"algo": "fullfact", "n_samples": 10}).
                If ``None``, do not perform calibration.
            **option_lists: Parameters for the clustering algorithm candidate.
                Each parameter has to be enclosed within a list.
                The list may contain different values to try out for the given
                parameter, or only one.
        """
        self.classif_cands.append(
            dict(
                name=name,
                calib_space=calib_space,
                calib_algo=calib_algo,
                **option_lists,
            )
        )

    def add_regressor_candidate(
        self,
        name: str,
        calib_space: DesignSpace | None = None,
        calib_algo: dict[str, str | int] | None = None,
        **option_lists: list[MLAlgoParameterType] | None,
    ) -> None:
        """Add a candidate for regression.

        Args:
            name: The name of a regression algorithm.
            calib_space: The space
                defining the calibration variables.
            calib_algo: The name and options of the DOE or optimization
                algorithm, e.g. {"algo": "fullfact", "n_samples": 10}).
                If ``None``, do not perform calibration.
            **option_lists: Parameters for the clustering algorithm candidate.
                Each parameter has to be enclosed within a list.
                The list may contain different values to try out for the given
                parameter, or only one.
        """
        self.regress_cands.append(
            dict(
                name=name,
                calib_space=calib_space,
                calib_algo=calib_algo,
                **option_lists,
            )
        )

    @DataFormatters.format_predict_class_dict
    @DataFormatters.format_samples
    @DataFormatters.format_transform(transform_outputs=False)
    def predict_class(
        self,
        input_data: DataType,
    ) -> int | ndarray:
        """Predict classes from input data.

        The user can specify these input data either as a NumPy array,
        e.g. ``array([1., 2., 3.])``
        or as a dictionary,
        e.g.  ``{'a': array([1.]), 'b': array([2., 3.])}``.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.

        Returns:
            The predicted classes.
        """
        return self.classifier.predict(input_data)

    @DataFormatters.format_input_output
    def predict_local_model(
        self,
        input_data: DataType,
        index: int,
    ) -> DataType:
        """Predict output data from input data.

        The user can specify these input data either as a NumPy array,
        e.g. ``array([1., 2., 3.])``
        or as a dictionary,
        e.g.  ``{'a': array([1.]), 'b': array([2., 3.])}``.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.
            index: The index of the local regression model.

        Returns:
            The predicted output data.
        """
        return self.regress_models[index].predict(input_data)

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
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
        _dataset = IODataset(dataset_name="training_set")
        _dataset.add_group(
            dataset.INPUT_GROUP,
            input_data,
            [self._LOCAL_INPUT],
            {self._LOCAL_INPUT: input_data.shape[1]},
        )
        _dataset.add_variable(
            self.LABELS, self.clusterer.labels[:, newaxis], _dataset.OUTPUT_GROUP
        )
        self._fit_classifier(_dataset)
        self._fit_regressors(dataset)

    def _fit_clusters(self, dataset: Dataset) -> None:
        """Train the clustering algorithm.

        The method adds resulting labels as a new output in the dataset.

        Args:
            dataset: The dataset containing input and output data.
        """
        if not self.cluster_cands:
            factory = ClusteringModelFactory()
            self.clusterer = factory.create(
                self.cluster_algo, data=dataset, **self.cluster_params
            )
            self.clusterer.learn()
        else:
            selector = MLAlgoSelection(
                dataset,
                self.cluster_measure["measure"],
                **self.cluster_measure["options"],
            )
            for cand in self.cluster_cands:
                selector.add_candidate(**cand)
            self.clusterer = selector.select()
            LOGGER.info("Selected clusterer:")
            with MultiLineString.offset():
                LOGGER.info("%s", self.clusterer)

    def _fit_classifier(self, dataset: IODataset) -> None:
        """Train the classification algorithm.

        Args:
            dataset: The dataset containing labeled input and output data.
        """
        if not self.classif_cands:
            factory = ClassificationModelFactory()
            self.classifier = factory.create(
                self.classif_algo,
                data=dataset,
                output_names=[self.LABELS],
                **self.classif_params,
            )
            self.classifier.learn()
        else:
            selector = MLAlgoSelection(
                dataset,
                self.classif_measure["measure"],
                **self.classif_measure["options"],
            )
            for cand in self.classif_cands:
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
        factory = RegressionModelFactory()
        self.regress_models = []
        for index in range(self.clusterer.n_clusters):
            samples = nonzero(self.clusterer.labels == index)[0]
            if not self.regress_cands:
                local_model = factory.create(
                    self.regress_algo, data=dataset, **self.regress_params
                )
                local_model.learn(samples=samples)
            else:
                selector = MLAlgoSelection(
                    dataset,
                    self.regress_measure["measure"],
                    samples=samples,
                    **self.regress_measure["options"],
                )
                for cand in self.regress_cands:
                    selector.add_candidate(**cand)
                local_model = selector.select()
                LOGGER.info("Selected regressor for cluster %s:", index)
                with MultiLineString.offset():
                    LOGGER.info("%s", local_model)

            self.regress_models.append(local_model)

    def _predict_all(
        self,
        input_data: ndarray,
    ) -> ndarray:
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
            self.regress_models[0].output_dimension,
        ))
        for i in range(self.n_clusters):
            output_data[:, i] = self.regress_models[i].predict(input_data)
        return output_data

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
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
        input_data: ndarray,
    ) -> ndarray:
        if self.hard:
            return self._predict_jacobian_hard(input_data)

        return self._predict_jacobian_soft(input_data)

    def _predict_jacobian_hard(
        self,
        input_data: ndarray,
    ) -> ndarray:
        """Predict the Jacobian matrices of the regression model at input_data.

        This method uses a hard classification.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The predicted Jacobian data with shape (n_samples, n_outputs, n_inputs).
        """
        classes = self.classifier.predict(input_data)[..., 0]
        first_regression_model = self.regress_models[0]
        jacobians = zeros((
            len(input_data),
            first_regression_model.output_dimension,
            first_regression_model.input_dimension,
        ))
        for klass in unique(classes):
            inds_kls = (classes == klass).nonzero()[0]
            jacobians[inds_kls] = self.regress_models[klass].predict_jacobian(
                input_data[inds_kls]
            )
        return jacobians

    def _predict_jacobian_soft(
        self,
        input_data: ndarray,
    ) -> NoReturn:
        """Predict the Jacobian matrices of the regression model at input_data.

        This method uses a soft classification.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The predicted Jacobian data with shape (n_samples, n_outputs, n_inputs).
        """
        raise NotImplementedError

    def _save_algo(
        self,
        directory: Path,
    ) -> None:
        self.clusterer.to_pickle(directory / "clusterer")
        self.classifier.to_pickle(directory / "classifier")
        for i, local_model in enumerate(self.regress_models):
            local_model.to_pickle(directory / f"local_model_{i}")

    def load_algo(  # noqa: D102
        self,
        directory: str | Path,
    ) -> None:
        directory = Path(directory)
        cluster_factory = ClusteringModelFactory()
        classif_factory = ClassificationModelFactory()
        regress_factory = RegressionModelFactory()
        self.clusterer = cluster_factory.load(directory / "clusterer")
        self.classifier = classif_factory.load(directory / "classifier")
        self.regress_models = []
        for i in range(self.n_clusters):
            self.regress_models.append(
                regress_factory.load(directory / f"local_model_{i}")
            )

    @property
    def __string_representation(self) -> MultiLineString:
        mls = super()._string_representation
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
        for i, local_model in enumerate(self.regress_models):
            mls.add("Local model {}", i)
            mls.indent()
            mls.add(str(local_model).split("\n")[0])
            mls.dedent()
        return mls

    def __repr__(self) -> str:
        return str(self.__string_representation)

    def _repr_html_(self) -> str:
        return self.__string_representation._repr_html_()

    def _get_objects_to_save(self) -> dict[str, SavedObjectType]:
        objects = super()._get_objects_to_save()
        objects["cluster_algo"] = self.cluster_algo
        objects["classif_algo"] = self.classif_algo
        objects["regress_algo"] = self.regress_algo
        objects["cluster_params"] = self.cluster_params
        objects["classif_params"] = self.classif_params
        objects["regress_params"] = self.regress_params
        return objects

    @property
    def labels(self) -> list[int]:
        """The cluster labels."""
        return self.clusterer.labels

    @property
    def n_clusters(self) -> int:
        """The number of clusters."""
        return self.clusterer.n_clusters
