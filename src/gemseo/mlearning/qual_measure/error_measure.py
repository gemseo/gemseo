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
"""Here is the baseclass to measure the error of machine learning algorithms.

The concept of error measure is implemented with the :class:`.MLErrorMeasure` class and
proposes different evaluation methods.
"""
from __future__ import division, unicode_literals

from copy import deepcopy
from typing import NoReturn, Optional, Sequence, Union

from numpy import arange
from numpy import delete as npdelete
from numpy import ndarray, unique
from numpy.random import choice

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.supervised import MLSupervisedAlgo
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure


class MLErrorMeasure(MLQualityMeasure):
    """An abstract error measure for machine learning."""

    def __init__(
        self,
        algo,  # type: MLSupervisedAlgo
    ):  # type: (...) -> None
        """
        Args:
            algo: A machine learning algorithm for supervised learning.
        """
        super(MLErrorMeasure, self).__init__(algo)

    def evaluate_learn(
        self,
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        samples = self._assure_samples(samples)
        self.algo.learn(samples)
        inputs = self.algo.input_data[samples]
        outputs = self.algo.output_data[samples]
        predictions = self.algo.predict(inputs)
        measure = self._compute_measure(outputs, predictions, multioutput)
        return measure

    def evaluate_test(
        self,
        test_data,  # type:Dataset
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        samples = self._assure_samples(samples)
        self.algo.learn(samples)
        in_grp = test_data.INPUT_GROUP
        out_grp = test_data.OUTPUT_GROUP
        inputs = test_data.get_data_by_group(in_grp)
        outputs = test_data.get_data_by_group(out_grp)
        predictions = self.algo.predict(inputs)
        measure = self._compute_measure(outputs, predictions, multioutput)
        return measure

    def evaluate_kfolds(
        self,
        n_folds=5,  # type: int
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
        randomize=False,  # type:bool
    ):  # type: (...) -> Union[float,ndarray]
        folds, samples = self._compute_folds(samples, n_folds, randomize)

        in_grp = self.algo.learning_set.INPUT_GROUP
        out_grp = self.algo.learning_set.OUTPUT_GROUP
        inputs = self.algo.learning_set.get_data_by_group(in_grp)
        outputs = self.algo.learning_set.get_data_by_group(out_grp)

        algo = deepcopy(self.algo)

        qualities = []
        for n_fold in range(n_folds):
            fold = folds[n_fold]
            train = npdelete(samples, fold)
            algo.learn(samples=train)
            expected = outputs[fold]
            predicted = algo.predict(inputs[fold])
            quality = self._compute_measure(expected, predicted, multioutput)
            qualities.append(quality)

        quality = sum(qualities) / len(qualities)

        return quality

    def evaluate_bootstrap(
        self,
        n_replicates=100,  # type: int
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        samples = self._assure_samples(samples)
        n_samples = samples.size
        inds = arange(n_samples)

        in_grp = self.algo.learning_set.INPUT_GROUP
        out_grp = self.algo.learning_set.OUTPUT_GROUP
        inputs = self.algo.learning_set.get_data_by_group(in_grp)
        outputs = self.algo.learning_set.get_data_by_group(out_grp)

        algo = deepcopy(self.algo)

        qualities = []
        for _ in range(n_replicates):
            train = unique(choice(n_samples, n_samples))
            test = npdelete(inds, train)
            algo.learn([samples[index] for index in train])
            test_samples = [samples[index] for index in test]
            expected = outputs[test_samples]
            predicted = algo.predict(inputs[test_samples])
            quality = self._compute_measure(expected, predicted, multioutput)
            qualities.append(quality)

        quality = sum(qualities) / len(qualities)

        return quality

    def _compute_measure(
        self,
        outputs,  # type: ndarray
        predictions,  # type: ndarray
        multioutput=True,  # type: bool
    ):  # type: (...) -> NoReturn
        """Compute the quality measure.

        Args:
            outputs: The reference data.
            predictions: The predicted labels.
            multioutput: Whether to return the quality measure
                for each output component. If not, average these measures.

        Returns:
            The value of the quality measure.
        """
        raise NotImplementedError
