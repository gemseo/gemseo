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
"""This module contains the base classes for clustering algorithms.

The :mod:`~gemseo.mlearning.cluster.cluster` module
implements the concept of clustering models,
a kind of unsupervised machine learning algorithm
where the goal is to group data into clusters.
Wherever possible,
these methods should be able to predict the class of the new data,
as well as the probability of belonging to each class.

This concept is implemented
through the :class:`.MLClusteringAlgo` class,
which inherits from the :class:`.MLUnsupervisedAlgo` class,
and through the :class:`.MLPredictiveClusteringAlgo` class
which inherits from :class:`.MLClusteringAlgo`.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Iterable
from typing import NoReturn
from typing import Sequence
from typing import Union

from numpy import atleast_2d
from numpy import ndarray
from numpy import unique
from numpy import zeros

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.ml_algo import SavedObjectType as MLAlgoSavedObjectType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.core.unsupervised import MLUnsupervisedAlgo
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array

SavedObjectType = Union[MLAlgoSavedObjectType, ndarray, int]


class MLClusteringAlgo(MLUnsupervisedAlgo):
    """Clustering algorithm.

    The inheriting classes shall overload the
    :meth:`!MLUnsupervisedAlgo._fit` method.
    """

    labels: list[int]
    """The indices of the clusters for the different samples."""

    n_clusters: int
    """The number of clusters."""

    def __init__(
        self,
        data: Dataset,
        transformer: TransformerType = MLUnsupervisedAlgo.IDENTITY,
        var_names: Iterable[str] | None = None,
        **parameters: MLAlgoParameterType,
    ) -> None:
        super().__init__(
            data, transformer=transformer, var_names=var_names, **parameters
        )
        self.labels = None
        self.n_clusters = None

    def _learn(
        self,
        indices: Sequence[int] | None,
        fit_transformers: bool,
    ) -> None:
        super()._learn(indices, fit_transformers=fit_transformers)
        if self.labels is None:
            raise ValueError("self._fit() shall assign labels.")
        self.n_clusters = unique(self.labels).shape[0]

    def _get_objects_to_save(self) -> dict[str, SavedObjectType]:
        objects = super()._get_objects_to_save()
        objects["labels"] = self.labels
        objects["n_clusters"] = self.n_clusters
        return objects


class MLPredictiveClusteringAlgo(MLClusteringAlgo):
    """Predictive clustering algorithm.

    The inheriting classes shall overload the
    :meth:`!MLUnsupervisedAlgo._fit` method, and the
    :meth:`!MLClusteringAlgo._predict` and
    :meth:`!MLClusteringAlgo._predict_proba` methods if possible.
    """

    def predict(
        self,
        data: DataType,
    ) -> int | ndarray:
        """Predict the clusters from the input data.

        The user can specify these input data either as a NumPy array,
        e.g. :code:`array([1., 2., 3.])`
        or as a dictionary,
        e.g.  :code:`{'a': array([1.]), 'b': array([2., 3.])}`.

        If the numpy arrays are of dimension 2,
        their i-th rows represent the input data of the i-th sample;
        while if the numpy arrays are of dimension 1,
        there is a single sample.

        The type of the output data and the dimension of the output arrays
        will be consistent
        with the type of the input data and the dimension of the input arrays.

        Args:
            data: The input data.

        Returns:
            The predicted cluster for each input data sample.
        """
        as_dict = isinstance(data, dict)
        if as_dict:
            data = concatenate_dict_of_arrays_to_array(data, self.var_names)
        single_sample = len(data.shape) == 1
        data = atleast_2d(data)
        parameters = self.learning_set.DEFAULT_GROUP
        if parameters in self.transformer:
            data = self.transformer[parameters].transform(data)
        clusters = self._predict(atleast_2d(data)).astype(int)
        if single_sample:
            clusters = clusters[0]
        return clusters

    @abstractmethod
    def _predict(
        self,
        data: ndarray,
    ) -> NoReturn:
        """Predict the clusters from input data.

        Args:
            data: The input data with the shape (n_samples, n_inputs).

        Returns:
            The predicted clusters with shape (n_samples,).
        """

    def predict_proba(
        self,
        data: DataType,
        hard: bool = True,
    ) -> ndarray:
        """Predict the probability of belonging to each cluster from input data.

        The user can specify these input data either as a numpy array,
        e.g. :code:`array([1., 2., 3.])`
        or as a dictionary,
        e.g.  :code:`{'a': array([1.]), 'b': array([2., 3.])}`.

        If the numpy arrays are of dimension 2,
        their i-th rows represent the input data of the i-th sample;
        while if the numpy arrays are of dimension 1,
        there is a single sample.

        The dimension of the output array
        will be consistent with the dimension of the input arrays.

        Args:
            data: The input data.
            hard: Whether clustering should be hard (True) or soft (False).

        Returns:
            The probability of belonging to each cluster,
            with shape (n_samples, n_clusters) or (n_clusters,).
        """
        as_dict = isinstance(data, dict)
        if as_dict:
            data = concatenate_dict_of_arrays_to_array(data, self.var_names)
        single_sample = len(data.shape) == 1
        data = atleast_2d(data)
        probas = self._predict_proba(atleast_2d(data), hard)
        if single_sample:
            probas = probas.ravel()
        return probas

    def _predict_proba(
        self,
        data: ndarray,
        hard: bool = True,
    ) -> ndarray:
        """Predict the probability of belonging to each cluster.

        Args:
            data: The input data with shape (n_samples, n_inputs).
            hard: Whether clustering should be hard (True) or soft (False).

        Returns:
            The probability of belonging to each cluster
                with shape (n_samples, n_clusters).
        """
        if hard:
            probas = self._predict_proba_hard(data)
        else:
            probas = self._predict_proba_soft(data)
        return probas

    def _predict_proba_hard(
        self,
        data: ndarray,
    ) -> ndarray:
        """Return 1 if the data belongs to a cluster, 0 otherwise.

        Args:
            data: The input data with shape (n_samples, n_inputs).

        Returns:
            The indicator of belonging to each cluster
                with shape (n_samples, n_clusters).
        """
        prediction = self._predict(data)
        probas = zeros((data.shape[0], self.n_clusters))
        for i, pred in enumerate(prediction):
            probas[i, pred] = 1
        return probas

    @abstractmethod
    def _predict_proba_soft(
        self,
        data: ndarray,
    ) -> NoReturn:
        """Predict the probability of belonging to each cluster.

        Args:
            The input data with shape (n_samples, n_inputs).

        Returns:
            The probability of belonging to each cluster
                with shape (n_samples, n_clusters).
        """
