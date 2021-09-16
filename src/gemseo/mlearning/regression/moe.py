# -*- coding: utf-8 -*-
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
"""The mixture of experts for regression.

The mixture of experts (MoE) regression model expresses the output
as a weighted sum of local surrogate models,
where the weights are indicating the class of the input.

Inputs are grouped into clusters by a classification model
that is trained on a training set
where the output labels are determined through a clustering algorithm.
The outputs may be preprocessed
through a sensor or a dimension reduction algorithm.

The classification may either be hard,
in which case only one of the weights is equal to one,
and the rest equal to zero:

.. math::

    y = \\sum_{k=1}^K i_{C_k}(x) f_k(x),

or soft,
in which case the weights express the probabilities of belonging to each class:

.. math::

    y = \\sum_{k=1}^K \\mathbb{P}(x\\in C_k) f_k(x),

where
:math:`x` is the input,
:math:`y` is the output,
:math:`K` is the number of classes,
:math:`(C_k)_{k=1,\\cdots,K}` are the input spaces associated to the classes,
:math:`i_{C_k}(x)` is the indicator of class :math:`k`,
:math:`\\mathbb{P}(x\\in C_k)` is the probability of class :math:`k`
given :math:`x` and
:math:`f_k(x)` is the local surrogate model on class :math:`k`.

This concept is implemented through the :class:`.MixtureOfExperts` class
which inherits from the :class:`.MLRegressionAlgo` class.
"""
from __future__ import division, unicode_literals

import logging
from typing import Callable, Dict, Iterable, List, NoReturn, Optional, Union

from numpy import ndarray, nonzero, unique, where, zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.core.dataset import Dataset
from gemseo.mlearning.classification.factory import ClassificationModelFactory
from gemseo.mlearning.cluster.factory import ClusteringModelFactory
from gemseo.mlearning.core.ml_algo import DataType, MLAlgoParameterType, TransformerType
from gemseo.mlearning.core.selection import MLAlgoSelection
from gemseo.mlearning.core.supervised import SavedObjectType
from gemseo.mlearning.qual_measure.f1_measure import F1Measure
from gemseo.mlearning.qual_measure.mse_measure import MSEMeasure
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure
from gemseo.mlearning.qual_measure.quality_measure import OptionType as EvalOptionType
from gemseo.mlearning.qual_measure.silhouette import SilhouetteMeasure
from gemseo.mlearning.regression.factory import RegressionModelFactory
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.py23_compat import Path
from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)

SavedObjectType = Union[SavedObjectType, str, Dict]

MLAlgoType = Dict[
    str,
    Optional[
        Union[str, DesignSpace, Dict[str, Union[str, int]], List[MLAlgoParameterType]]
    ],
]


class MixtureOfExperts(MLRegressionAlgo):
    """Mixture of experts regression.

    Attributes:
        hard (bool): Whether clustering/classification should be hard or soft.
        cluster_algo (str): The name of the clustering algorithm.
        classif_algo (str): The name of the classification algorithm.
        regress_algo (str): The name of the regression algorithm.
        cluster_params (Optional[MLAlgoParameterType]): The parameters
            of the clustering algorithm.
        classif_params (Optional[MLAlgoParameterType]): The parameters
            of the classification algorithm.
        regress_params (Optional[MLAlgoParameterType]): The parameters
            of the regression algorithm.
        cluster_measure (Dict[str,Union[str,EvalOptionType]]): The quality measure
            for the clustering algorithms.
        classif_measure (Dict[str,Union[str,EvalOptionType]]): The quality measure
            for the classification algorithms.
        regress_measure (Dict[str,Union[str,EvalOptionType]]): The quality measure
            for the regression algorithms.
        cluster_cands (List[MLAlgoType]): The clustering algorithm candidates.
        classif_cands (List[MLAlgoType]): The classification algorithm candidates.
        regress_cands (List[MLAlgoType]): The regression algorithm candidates.
        clusterer (MLClusteringAlgo): The clustering algorithm.
        classifier (MLClassificationAlgo): The classification algorithm.
        regress_models (List(MLRegressionAlgo)): The regression algorithms.
    """

    ABBR = "MoE"

    LABELS = "labels"

    _LOCAL_INPUT = "input"
    _LOCAL_OUTPUT = "output"

    def __init__(
        self,
        data,  # type: Dataset
        transformer=None,  # type: Optional[TransformerType]
        input_names=None,  # type: Optional[Iterable[str]]
        output_names=None,  # type: Optional[Iterable[str]]
        hard=True,  # type: bool
    ):  # type: (...) -> None
        """
        Args:
            hard: Whether clustering/classification should be hard or soft.
        """
        super(MixtureOfExperts, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            hard=hard,
        )
        self.hard = hard
        self.cluster_algo = "KMeans"
        self.classif_algo = "KNNClassifier"
        self.regress_algo = "LinearRegression"
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

    class DataFormatters(MLRegressionAlgo.DataFormatters):
        """Machine learning regression model decorators."""

        @classmethod
        def format_predict_class_dict(
            cls,
            predict,  # type: Callable[[ndarray],ndarray]
        ):  # type: (...) -> Callable[[DataType],DataType]
            """Make an array-based function be called with a dictionary of NumPy arrays.

            Args:
                predict: The function to be called;
                    it takes a NumPy array in input and returns a NumPy array.

            Returns:
                A function making the function 'predict' work with
                either a NumPy data array
                or a dictionary of NumPy data arrays indexed by variables names.
                The evaluation will have the same type as the input data.
            """

            def wrapper(
                self,
                input_data,  # type: DataType
                *args,
                **kwargs
            ):  # type: (...) -> DataType
                """Evaluate 'predict' with either array or dictionary-based input data.

                Firstly,
                the pre-processing stage converts the input data to a NumPy data array,
                if these data are expressed as a dictionary of NumPy data arrays.

                Then,
                the processing evaluates the function 'predict'
                from this NumPy input data array.

                Lastly,
                the post-processing transforms the output data
                to a dictionary of output NumPy data array
                if the input data were passed as a dictionary of NumPy data arrays.

                Args:
                    input_data: The input data.
                    *args: The positional arguments of the function 'predict'.
                    **kwargs: The keyword arguments of the function 'predict'.

                Returns:
                    The output data with the same type as the input one.
                """
                as_dict = isinstance(input_data, dict)
                if as_dict:
                    input_data = DataConversion.dict_to_array(
                        input_data, self.input_names
                    )
                output_data = predict(self, input_data, *args, **kwargs)
                if as_dict:
                    output_data = {self.LABELS: output_data}
                return output_data

            return wrapper

    def set_clusterer(
        self,
        cluster_algo,  # type: str
        **cluster_params  # type:Optional[MLAlgoParameterType]
    ):  # type: (...) -> None
        """Set the clustering algorithm.

        Args:
            cluster_algo: The name of the clustering algorithm.
            **cluster_params: The parameters of the clustering algorithm.
        """
        self.cluster_algo = cluster_algo
        self.cluster_params = cluster_params

    def set_classifier(
        self,
        classif_algo,  # type: str
        **classif_params  # type:Optional[MLAlgoParameterType]
    ):  # type: (...) -> None
        """Set the classification algorithm.

        Args:
            classif_algo: The name of the classification algorithm.
            **classif_params: The parameters of the classification algorithm.
        """
        self.classif_algo = classif_algo
        self.classif_params = classif_params

    def set_regressor(
        self,
        regress_algo,  # type: str
        **regress_params  # type:Optional[MLAlgoParameterType]
    ):  # type: (...) -> None
        """Set the regression algorithm.

        Args:
            regress_algo: The name of the regression algorithm.
            **regress_params: The parameters of the regression algorithm.
        """
        self.regress_algo = regress_algo
        self.regress_params = regress_params

    def set_clustering_measure(
        self,
        measure,  # type: MLQualityMeasure
        **eval_options  # type: EvalOptionType
    ):  # type: (...) -> None
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
        measure,  # type: MLQualityMeasure
        **eval_options  # type: EvalOptionType
    ):  # type: (...) -> None
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
        measure,  # type: MLQualityMeasure
        **eval_options  # type: EvalOptionType
    ):  # type: (...) -> None
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
        name,  # type: str
        calib_space=None,  # type: Optional[DesignSpace]
        calib_algo=None,  # type: Optional[Dict[str,Union[str,int]]]
        **option_lists  # type:Optional[List[MLAlgoParameterType]]
    ):  # type: (...) -> None
        """Add a candidate for clustering.

        Args:
            name: The name of a clustering algorithm.
            calib_space: The space
                defining the calibration variables.
            calib_algo: The name and options of the DOE or optimization
                algorithm, e.g. {"algo": "fullfact", "n_samples": 10}).
                If None, do not perform calibration.
            *** option_lists: Parameters for the clustering algorithm candidate.
                Each parameter has to be enclosed within a list.
                The list may contain different values to try out for the given
                parameter, or only one.
        """
        self.cluster_cands.append(
            dict(
                name=name,
                calib_space=calib_space,
                calib_algo=calib_algo,
                **option_lists
            )
        )

    def add_classifier_candidate(
        self,
        name,  # type: str
        calib_space=None,  # type: Optional[DesignSpace]
        calib_algo=None,  # type: Optional[Dict[str,Union[str,int]]]
        **option_lists  # type:Optional[List[MLAlgoParameterType]]
    ):  # type: (...) -> None
        """Add a candidate for classification.

        Args:
            name: The name of a classification algorithm.
            calib_space: The space
                defining the calibration variables.
            calib_algo: The name and options of the DOE or optimization
                algorithm, e.g. {"algo": "fullfact", "n_samples": 10}).
                If None, do not perform calibration.
            *** option_lists: Parameters for the clustering algorithm candidate.
                Each parameter has to be enclosed within a list.
                The list may contain different values to try out for the given
                parameter, or only one.
        """
        self.classif_cands.append(
            dict(
                name=name,
                calib_space=calib_space,
                calib_algo=calib_algo,
                **option_lists
            )
        )

    def add_regressor_candidate(
        self,
        name,  # type: str
        calib_space=None,  # type: Optional[DesignSpace]
        calib_algo=None,  # type: Optional[Dict[str,Union[str,int]]]
        **option_lists  # type:Optional[List[MLAlgoParameterType]]
    ):  # type: (...) -> None
        """Add a candidate for regression.

        Args:
            name: The name of a regression algorithm.
            calib_space: The space
                defining the calibration variables.
            calib_algo: The name and options of the DOE or optimization
                algorithm, e.g. {"algo": "fullfact", "n_samples": 10}).
                If None, do not perform calibration.
            *** option_lists: Parameters for the clustering algorithm candidate.
                Each parameter has to be enclosed within a list.
                The list may contain different values to try out for the given
                parameter, or only one.
        """
        self.regress_cands.append(
            dict(
                name=name,
                calib_space=calib_space,
                calib_algo=calib_algo,
                **option_lists
            )
        )

    @DataFormatters.format_predict_class_dict
    @DataFormatters.format_samples
    @DataFormatters.format_transform(transform_outputs=False)
    def predict_class(
        self,
        input_data,  # type: DataType
    ):  # type: (...) -> Union[int,ndarray]
        """Predict classes from input data.

        The user can specify these input data either as a NumPy array,
        e.g. :code:`array([1., 2., 3.])`
        or as a dictionary,
        e.g.  :code:`{'a': array([1.]), 'b': array([2., 3.])}`.

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
        input_data,  # type: DataType
        index,  # type: int
    ):  # type: (...) -> DataType
        """Predict output data from input data.

        The user can specify these input data either as a NumPy array,
        e.g. :code:`array([1., 2., 3.])`
        or as a dictionary,
        e.g.  :code:`{'a': array([1.]), 'b': array([2., 3.])}`.

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
        input_data,  # type: ndarray
        output_data,  # type: ndarray
    ):  # type: (...) -> None
        dataset = Dataset("training_set")
        dataset.add_group(
            Dataset.INPUT_GROUP,
            input_data,
            [self._LOCAL_INPUT],
            {self._LOCAL_INPUT: input_data.shape[1]},
        )
        dataset.add_group(
            Dataset.OUTPUT_GROUP,
            output_data,
            [self._LOCAL_OUTPUT],
            {self._LOCAL_OUTPUT: output_data.shape[1]},
            cache_as_input=False,
        )
        self._fit_clusters(dataset)
        self._fit_classifier(dataset)
        self._fit_regressors(dataset)

    def _fit_clusters(
        self,
        dataset,  # type:Dataset
    ):  # type: (...) -> None
        """Train the clustering algorithm.

        The methods adds resulting labels as a new output in the dataset.

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
                **self.cluster_measure["options"]
            )
            for cand in self.cluster_cands:
                selector.add_candidate(**cand)
            self.clusterer = selector.select()
            LOGGER.info("Selected clusterer:")
            with MultiLineString.offset():
                LOGGER.info("%s", self.clusterer)

        labels = self.clusterer.labels[:, None]
        dataset.add_variable(self.LABELS, labels, self.LABELS, False)

    def _fit_classifier(
        self,
        dataset,  # type:Dataset
    ):  # type: (...) -> None
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
                **self.classif_params
            )
            self.classifier.learn()
        else:
            selector = MLAlgoSelection(
                dataset,
                self.classif_measure["measure"],
                **self.classif_measure["options"]
            )
            for cand in self.classif_cands:
                selector.add_candidate(output_names=[[self.LABELS]], **cand)
            self.classifier = selector.select()
            LOGGER.info("Selected classifier:")
            with MultiLineString.offset():
                LOGGER.info("%s", self.classifier)

    def _fit_regressors(
        self,
        dataset,  # type:Dataset
    ):  # type: (...) -> None
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
                    **self.regress_measure["options"]
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
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Predict output of each regression model for given input data.

        This method stacks the different outputs along a new axis.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The output data with shape (n_samples, n_clusters, n_outputs).
        """
        # dim(input_data)  = (n_samples, n_inputs)
        # dim(output_data) = (n_samples, n_clusters, n_outputs)
        output_data = zeros(
            (input_data.shape[0], self.n_clusters, self.regress_models[0].output_shape)
        )
        for i in range(self.n_clusters):
            output_data[:, i] = self.regress_models[i].predict(input_data)
        return output_data

    def _predict(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        # dim(probas)         = (n_samples, n_clusters,     1    )
        # dim(local_outputs)  = (n_samples, n_clusters, n_outputs)
        # dim(contributions)  = (n_samples, n_clusters, n_outputs)
        # dim(global_outputs) = (n_samples, n_outputs)
        probas = self.classifier.predict_proba(input_data, hard=self.hard)
        local_outputs = self._predict_all(input_data)
        contributions = probas * local_outputs
        global_outputs = contributions.sum(axis=1)
        return global_outputs

    def _predict_jacobian(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        if self.hard:
            jacobians = self._predict_jacobian_hard(input_data)
        else:
            jacobians = self._predict_jacobian_soft(input_data)
        return jacobians

    def _predict_jacobian_hard(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Predict the Jacobian matrices of the regression model at input_data.

        This method uses a hard classification.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The predicted Jacobian data with shape (n_samples, n_outputs, n_inputs).
        """
        n_samples = input_data.shape[0]
        classes = self.classifier.predict(input_data)[..., 0]
        unq_classes = unique(classes)
        jacobians = zeros(
            (
                n_samples,
                self.regress_models[0].output_shape,
                self.regress_models[0].input_shape,
            )
        )
        for klass in unq_classes:
            inds_kls = where(classes == klass)
            jacobians[inds_kls] = self.regress_models[klass].predict_jacobian(
                input_data[inds_kls]
            )
        return jacobians

    def _predict_jacobian_soft(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> NoReturn
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
        directory,  # type: Path
    ):  # type: (...) -> None
        self.clusterer.save(directory / "clusterer")
        self.classifier.save(directory / "classifier")
        for i, local_model in enumerate(self.regress_models):
            local_model.save(directory / "local_model_{}".format(i))

    def load_algo(
        self,
        directory,  # type: Union[str,Path]
    ):  # type: (...) -> None
        directory = Path(directory)
        cluster_factory = ClusteringModelFactory()
        classif_factory = ClassificationModelFactory()
        regress_factory = RegressionModelFactory()
        self.clusterer = cluster_factory.load(directory / "clusterer")
        self.classifier = classif_factory.load(directory / "classifier")
        self.regress_models = []
        for i in range(self.n_clusters):
            self.regress_models.append(
                regress_factory.load(directory / "local_model_{}".format(i))
            )

    def __str__(self):  # type: (...) -> str
        string = MultiLineString()
        string.add(super(MixtureOfExperts, self).__str__())
        string.indent()
        string.indent()
        string.add("Clustering")
        string.indent()
        string.add(str(self.clusterer).split("\n")[0])
        string.dedent()
        string.add("Classification")
        string.indent()
        string.add(str(self.classifier).split("\n")[0])
        string.dedent()
        string.add("Regression")
        string.indent()
        for i, local_model in enumerate(self.regress_models):
            string.add("Local model {}", i)
            string.indent()
            string.add(str(local_model).split("\n")[0])
            string.dedent()
        return str(string)

    def _get_objects_to_save(self):  # type: (...) -> Dict[str,SavedObjectType]
        objects = super(MixtureOfExperts, self)._get_objects_to_save()
        objects["cluster_algo"] = self.cluster_algo
        objects["classif_algo"] = self.classif_algo
        objects["regress_algo"] = self.regress_algo
        objects["cluster_params"] = self.cluster_params
        objects["classif_params"] = self.classif_params
        objects["regress_params"] = self.regress_params
        return objects

    @property
    def labels(self):  # type:(...) -> List[int]
        """The cluster labels."""
        return self.clusterer.labels

    @property
    def n_clusters(self):  # type:(...) -> int
        """The number of clusters."""
        return self.clusterer.n_clusters
