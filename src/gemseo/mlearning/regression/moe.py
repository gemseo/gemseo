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
from __future__ import absolute_import, division, unicode_literals

import logging
from os.path import join

from numpy import nonzero, unique, where, zeros

from gemseo.core.dataset import Dataset
from gemseo.mlearning.classification.factory import ClassificationModelFactory
from gemseo.mlearning.cluster.factory import ClusteringModelFactory
from gemseo.mlearning.core.selection import MLAlgoSelection
from gemseo.mlearning.qual_measure.f1_measure import F1Measure
from gemseo.mlearning.qual_measure.mse_measure import MSEMeasure
from gemseo.mlearning.qual_measure.silhouette import SilhouetteMeasure
from gemseo.mlearning.regression.factory import RegressionModelFactory
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.string_tools import MultiLineString

"""
Mixture of Experts
==================

The mixture of experts (MoE) regression model expresses the output as a
weighted sum of local surrogate models, where the weights are indicating
the class of the input.

Inputs are grouped into clusters by a classification model that is trained on a
training set where the output labels are determined through a clustering
algorithm. The outputs may be preprocessed trough a sensor or a dimension
reduction algorithm.

The classification may either be hard, in which only one of the weights is
equal to one, and the rest equal to zero:

.. math::

    y = \\sum_{k=1}^K i_{C_k}(x) f_k(x),

or soft, in which case the weights express the probabilities of belonging to
each class:

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

This concept is implemented through the :class:`.MixtureOfExperts` class which
inherits from the :class:`.MLRegressionAlgo` class.
"""


LOGGER = logging.getLogger(__name__)


class MixtureOfExperts(MLRegressionAlgo):
    """Mixture of experts regression."""

    ABBR = "MoE"

    LABELS = "labels"

    _LOCAL_INPUT = "input"
    _LOCAL_OUTPUT = "output"

    def __init__(
        self,
        data,
        transformer=None,
        input_names=None,
        output_names=None,
        hard=True,
    ):
        """Constructor.

        :param data: learning dataset.
        :type data: Dataset
        :param transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :type transformer: dict(str)
        :param input_names: names of the input variables.
        :type input_names: list(str)
        :param output_names: names of the output variables.
        :type output_names: list(str)
        :param hard: Indicator for hard or soft clustering/classification.
            Hard clustering/classification if True. Default: True.
        :type hard: bool
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
        def format_predict_class_dict(cls, predict):
            """If input_data is passed as a dictionary, then convert it to ndarray, and
            convert output_data to dictionary. Else, do nothing.

            :param predict: Method whose input_data and output_data are to be
                formatted.
            """

            def wrapper(self, input_data, *args, **kwargs):
                """Wrapper function."""
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

    def set_clusterer(self, cluster_algo, **cluster_params):
        """Set cluster algorithm.

        :param str cluster_algo: clusterer.
        :param cluster_params: optional arguments for clustering.
            If none, uses default arguments.
        """
        self.cluster_algo = cluster_algo
        self.cluster_params = cluster_params

    def set_classifier(self, classif_algo, **classif_params):
        """Set classification algorithm.

        :param str classif_algo: classifier.
        :param classif_params: optional arguments for classification.
            If none, uses default arguments.
        """
        self.classif_algo = classif_algo
        self.classif_params = classif_params

    def set_regressor(self, regress_algo, **regress_params):
        """Set regression algorithm.

        :param str regress_algo: regressor.
        :param regress_params: optional arguments for regression.
            If none, uses default arguments.
        """
        self.regress_algo = regress_algo
        self.regress_params = regress_params

    def set_clustering_measure(self, measure, **eval_options):
        """Set quality measure for the clusterer.

        :param MLQualityMeasure measure: clustering quality measure.
        :param options: options to pass to the quality measure evaluation.
        """
        self.cluster_measure = {
            "measure": measure,
            "options": eval_options,
        }

    def set_classification_measure(self, measure, **eval_options):
        """Set quality measure for the classifier.

        :param MLQualityMeasure measure: classification quality measure.
        :param options: options to pass to the quality measure evaluation.
        """
        self.classif_measure = {
            "measure": measure,
            "options": eval_options,
        }

    def set_regression_measure(self, measure, **eval_options):
        """Set quality measure for the regressors.

        :param MLQualityMeasure measure: regression quality measure.
        :param options: options to pass to the quality measure evaluation.
        """
        self.regress_measure = {
            "measure": measure,
            "options": eval_options,
        }

    def add_clusterer_candidate(
        self, name, calib_space=None, calib_algo=None, **option_lists
    ):
        """Add candidate for clustering algorithm.

        :param str name: name of clustering algorithm.
        :param DesignSpace calib_space: Design space for parameters to be
            calibrated with an MLAlgoCalibration. If None, do not perform
            calibration. Default: None.
        :param dict calib_algo: Dictionary containing optimization algorithm
            and parameters (example: {"algo": "fullfact", "n_samples": 10}).
            If None, do not perform calibration. Default: None.
        :param dict option_lists: Parameters for the clustering algorithm
            candidate. Each parameter has to be enclosed within a list. The
            list may contain different values to try out for the given
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
        self, name, calib_space=None, calib_algo=None, **option_lists
    ):
        """Add candidate for classification algorithm.

        :param str name: name of classification algorithm.
        :param DesignSpace calib_space: Design space for parameters to be
            calibrated with an MLAlgoCalibration. If None, do not perform
            calibration. Default: None.
        :param dict calib_algo: Dictionary containing optimization algorithm
            and parameters (example: {"algo": "fullfact", "n_samples": 10}).
            If None, do not perform calibration. Default: None.
        :param dict option_lists: Parameters for the classification algorithm
            candidate. Each parameter has to be enclosed within a list. The
            list may contain different values to try out for the given
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
        self, name, calib_space=None, calib_algo=None, **option_lists
    ):
        """Add candidate for regression algorithm.

        :param str name: name of regression algorithm.
        :param DesignSpace calib_space: Design space for parameters to be
            calibrated with an MLAlgoCalibration. If None, do not perform
            calibration. Default: None.
        :param dict calib_algo: Dictionary containing optimization algorithm
            and parameters (example: {"algo": "fullfact", "n_samples": 10}).
            If None, do not perform calibration. Default: None.
        :param dict option_lists: Parameters for the regression algorithm
            candidate. Each parameter has to be enclosed within a list. The
            list may contain different values to try out for the given
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
    def predict_class(self, input_data):
        """Predict classes of input data.

        :param input_data: input data (1D or 2D).
        :type input_data: dict(ndarray) or ndarray
        :return: output classes ("0D" or 1D, one less than input data).
        :rtype: int or ndarray(int)
        """
        return self.classifier.predict(input_data)

    @DataFormatters.format_input_output
    def predict_local_model(self, input_data, index):
        """Predict output for given input data, using an individual regression model
        from the list.

        :param input_data: input data (1D or 2D).
        :type input_data: dict(ndarray) or ndarray
        :param int index: index of the local regression model.
        :return: output data (1D or 2D, same as input_data).
        :rtype: dict(ndarray) or ndarray
        """
        return self.regress_models[index].predict(input_data)

    def _fit(self, input_data, output_data):
        """Fit the regression model. As the data is provided as two numpy arrays, we
        construct a temporary dataset in order to use clustering, classification and
        regression algorithms.

        :param ndarray input_data: input data (2D).
        :param ndarray output_data: output data (2D).
        """
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

    def _fit_clusters(self, dataset):
        """Fit the clustering algorithm to the dataset (input/output labels ignored by
        clustering algorithm). Add resulting labels as a new output in the dataset.

        :param Dataset dataset: dataset containing input and output data.
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

    def _fit_classifier(self, dataset):
        """Fit the input data to the labels using the given classification algorithm.

        :param Dataset dataset: dataset containing input and output data,
            as well as labels after execution of _fit_clusters().
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

    def _fit_regressors(self, dataset):
        """Fit the local regression models on each cluster separately.

        :param Dataset dataset: dataset containing input and output data.
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

    def _predict_all(self, input_data):
        """Predict output of each regression model for given input data. Stack the
        different outputs along a new axis.

        :param ndarray input_data: input data (2D).
        :return: all output predictions (3D).
        :rtype: ndarray
        """
        # dim(input_data)  = (n_samples, n_inputs)
        # dim(output_data) = (n_samples, n_clusters, n_outputs)
        output_data = zeros(
            (input_data.shape[0], self.n_clusters, self.regress_models[0].output_shape)
        )
        for i in range(self.n_clusters):
            output_data[:, i] = self.regress_models[i].predict(input_data)
        return output_data

    def _predict(self, input_data):
        """Predict global output for given input data. The global output is computed as
        a sum of contributions from the individual local regression models, weighted by
        the probabilities of belonging to each cluster.

        :param ndarray input_data: input data (2D).e
        :return: global output prediction (2D).
        :rtype: ndarray
        """
        # dim(probas)         = (n_samples, n_clusters,     1    )
        # dim(local_outputs)  = (n_samples, n_clusters, n_outputs)
        # dim(contributions)  = (n_samples, n_clusters, n_outputs)
        # dim(global_outputs) = (n_samples, n_outputs)
        probas = self.classifier.predict_proba(input_data, hard=self.hard)
        local_outputs = self._predict_all(input_data)
        contributions = probas * local_outputs
        global_outputs = contributions.sum(axis=1)
        return global_outputs

    def _predict_jacobian(self, input_data):
        """Predict Jacobian of the regression model for the given input data.

        :param ndarray input_data: input_data (2D).
        :return: Jacobian matrices (3D, one for each sample).
        :rtype: ndarray
        """
        if self.hard:
            jacobians = self._predict_jacobian_hard(input_data)
        else:
            jacobians = self._predict_jacobian_soft(input_data)
        return jacobians

    def _predict_jacobian_hard(self, input_data):
        """Predict Jacobian of the regression model for the given input data, with a
        hard (constant) classification.

        :param ndarray input_data: input_data (2D).
        :return: Jacobian matrices (3D, one 2D matrix for each sample).
        :rtype: ndarray
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

    def _predict_jacobian_soft(self, input_data):
        """Predict Jacobian of the regression model for the given input data, with a
        soft classification.

        :param ndarray input_data: input_data (2D).
        :return: Jacobian matrices (3D, one 2D matrix for each sample).
        :rtype: ndarray
        """
        raise NotImplementedError

    def _save_algo(self, directory):
        """Save external machine learning algorithm.

        :param str directory: algorithm directory.
        """
        self.clusterer.save(join(directory, "clusterer"))
        self.classifier.save(join(directory, "classifier"))
        for i, local_model in enumerate(self.regress_models):
            local_model.save(join(directory, "local_model_{}".format(i)))

    def load_algo(self, directory):
        """Load external machine learning algorithm.

        :param str directory: algorithm directory.
        """
        cluster_factory = ClusteringModelFactory()
        classif_factory = ClassificationModelFactory()
        regress_factory = RegressionModelFactory()
        self.clusterer = cluster_factory.load(join(directory, "clusterer"))
        self.classifier = classif_factory.load(join(directory, "classifier"))
        self.regress_models = []
        for i in range(self.n_clusters):
            self.regress_models.append(
                regress_factory.load(join(directory, "local_model_{}".format(i)))
            )

    def __str__(self):
        """String representation for end user."""
        string = super(MixtureOfExperts, self).__str__()
        string += "\n\nClusterer:\n{}".format(self.clusterer)
        string += "\n\nClassifier:\n{}".format(self.classifier)
        for i, local_model in enumerate(self.regress_models):
            string += "\n\nLocal model {}:\n{}".format(i, local_model)
        return string

    def _get_objects_to_save(self):
        """Get objects to save.

        :return: objects to save.
        :rtype: dict
        """
        objects = super(MixtureOfExperts, self)._get_objects_to_save()
        objects["cluster_algo"] = self.cluster_algo
        objects["classif_algo"] = self.classif_algo
        objects["regress_algo"] = self.regress_algo
        objects["cluster_params"] = self.cluster_params
        objects["classif_params"] = self.classif_params
        objects["regress_params"] = self.regress_params
        return objects

    @property
    def labels(self):
        """Cluster labels."""
        return self.clusterer.labels

    @property
    def n_clusters(self):
        """Number of clusters."""
        return self.clusterer.n_clusters
