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
"""Here is the baseclass to measure the quality of machine learning algorithms.

The concept of clustering quality measure is implemented with the
:class:`.MLClusteringMeasure` class and proposes different evaluation methods.
"""
from __future__ import division
from __future__ import unicode_literals

from copy import deepcopy
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Union

from numpy import arange
from numpy import delete as npdelete
from numpy import ndarray
from numpy import unique
from numpy.random import choice

from gemseo.core.dataset import Dataset
from gemseo.mlearning.cluster.cluster import MLClusteringAlgo
from gemseo.mlearning.cluster.cluster import MLPredictiveClusteringAlgo
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure


class MLClusteringMeasure(MLQualityMeasure):
    """An abstract clustering measure for clustering algorithms."""

    def __init__(
        self,
        algo,  # type: MLClusteringAlgo
    ):  # type: (...) -> None
        """
        Args:
            algo: A machine learning algorithm for clustering.
        """
        super(MLClusteringMeasure, self).__init__(algo)

    def evaluate_learn(
        self,
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        samples = self._assure_samples(samples)
        if not self.algo.is_trained:
            self.algo.learn(samples)

        return self._compute_measure(
            self._get_data()[samples], self.algo.labels, multioutput
        )

    def _compute_measure(
        self,
        data,  # type: ndarray
        labels,  # type: ndarray
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        """Compute the quality measure.

        Args:
            data: The reference data.
            labels: The predicted labels.
            multioutput: Whether to return the quality measure
                for each output component; if not, average these measures.

        Returns:
            The value of the quality measure.
        """
        raise NotImplementedError

    def _get_data(self):  # type: (...) -> Dict[str,ndarray]
        """Get data.

        Returns:
            The learning data indexed by the names of the variables.
        """
        return self.algo.learning_set.get_data_by_names(self.algo.var_names, False)


class MLPredictiveClusteringMeasure(MLClusteringMeasure):
    """An abstract clustering measure for predictive clustering algorithms."""

    def __init__(
        self,
        algo,  # type: MLPredictiveClusteringAlgo
    ):  # type: (...) -> None
        """
        Args:
            algo: A machine learning algorithm for predictive clustering.
        """
        super(MLPredictiveClusteringMeasure, self).__init__(algo)

    def evaluate_test(
        self,
        test_data,  # type:Dataset
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        samples = self._assure_samples(samples)
        if not self.algo.is_trained:
            self.algo.learn(samples)

        data = test_data.get_data_by_names(self.algo.var_names, False)
        return self._compute_measure(data, self.algo.predict(data), multioutput)

    def evaluate_kfolds(
        self,
        n_folds=5,  # type: int
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
        randomize=False,  # type:bool
        seed=None,  # type: Optional[int]
    ):  # type: (...) -> Union[float,ndarray]
        folds, samples = self._compute_folds(samples, n_folds, randomize, seed)

        data = self._get_data()

        algo = deepcopy(self.algo)

        qualities = []
        for n_fold in range(n_folds):
            test_indices = folds[n_fold]
            algo.learn(samples=npdelete(samples, test_indices))
            test_data = data[test_indices]
            qualities.append(
                self._compute_measure(test_data, algo.predict(test_data), multioutput)
            )

        return sum(qualities) / len(qualities)

    def evaluate_bootstrap(
        self,
        n_replicates=100,  # type: int
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        samples = self._assure_samples(samples)
        n_samples = samples.size
        indices = arange(n_samples)

        data = self._get_data()

        algo = deepcopy(self.algo)

        qualities = []
        for _ in range(n_replicates):
            training_indices = unique(choice(n_samples, n_samples))
            algo.learn(samples=[samples[index] for index in training_indices])
            test_data = data[
                [samples[index] for index in npdelete(indices, training_indices)]
            ]
            qualities.append(
                self._compute_measure(test_data, algo.predict(test_data), multioutput)
            )

        return sum(qualities) / len(qualities)

    def _compute_measure(
        self,
        data,  # type: ndarray
        labels,  # type: ndarray
        multioutput=True,  # type: bool
    ):  # type: (...) -> NoReturn
        """Compute the quality measure.

        Args:
            data: The reference data.
            labels: The predicted labels.
            multioutput: Whether to return the quality measure
                for each output component. If not, average these measures.

        Returns:
            The value of the quality measure.
        """
        raise NotImplementedError

    def _get_data(self):  # type: (...) -> Dict[str,ndarray]
        """Get data.

        Returns:
            The learning data indexed by the names of the variables.
        """
        return self.algo.learning_set.get_data_by_names(self.algo.var_names, False)
