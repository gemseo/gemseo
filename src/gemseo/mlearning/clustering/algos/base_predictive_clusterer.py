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
"""The base class for clustering algorithms with a prediction method."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import NoReturn

from numpy import atleast_2d
from numpy import ndarray
from numpy import zeros

from gemseo.mlearning.clustering.algos.base_clusterer import BaseClusterer
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array

if TYPE_CHECKING:
    from gemseo.mlearning.core.algos.ml_algo import DataType


class BasePredictiveClusterer(BaseClusterer):
    """The base class for clustering algorithms with a prediction method."""

    def predict(
        self,
        data: DataType,
    ) -> int | ndarray:
        """Predict the clusters from the input data.

        The user can specify these input data either as a NumPy array,
        e.g. ``array([1., 2., 3.])``
        or as a dictionary,
        e.g.  ``{'a': array([1.]), 'b': array([2., 3.])}``.

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
        if isinstance(data, Mapping):
            data = concatenate_dict_of_arrays_to_array(data, self.var_names)

        data_2d = atleast_2d(data)
        parameters = self.learning_set.DEFAULT_GROUP
        if parameters in self.transformer:
            data_2d = self.transformer[parameters].transform(data_2d)

        clusters = self._predict(data_2d).astype(int)
        if data.ndim == 1:
            return clusters[0]

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
        e.g. ``array([1., 2., 3.])``
        or as a dictionary,
        e.g.  ``{'a': array([1.]), 'b': array([2., 3.])}``.

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
        if isinstance(data, Mapping):
            data = concatenate_dict_of_arrays_to_array(data, self.var_names)

        probas = self._predict_proba(atleast_2d(data), hard)
        if data.ndim == 1:
            return probas.ravel()

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
            return self._predict_proba_hard(data)

        return self._predict_proba_soft(data)

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
            data: The input data with shape (n_samples, n_inputs).

        Returns:
            The probability of belonging to each cluster
                with shape (n_samples, n_clusters).
        """
