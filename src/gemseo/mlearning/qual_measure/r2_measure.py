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
"""
R2 error measure
================

The :mod:`~gemseo.mlearning.qual_measure.r2_measure` module
implements the concept of R2 measures for machine learning algorithms.

This concept is implemented through the :class:`.R2Measure` class
and overloads the :meth:`!MLErrorMeasure._compute_measure` method.

The R2 is defined by

.. math::

    R_2(\\hat{y}) = 1 - \\frac{\\sum_i (\\hat{y}_i - y_i)^2}
                              {\\sum_i (y_i-\\bar{y})^2},

where
:math:`\\hat{y}` are the predictions,
:math:`y` are the data points and
:math:`\\bar{y}` is the mean of :math:`y`.
"""
from __future__ import absolute_import, division, unicode_literals

from numpy import array_split, atleast_2d
from numpy import delete as npdelete
from numpy import mean, repeat
from sklearn.metrics import mean_squared_error, r2_score

from gemseo.mlearning.qual_measure.error_measure import MLErrorMeasure


class R2Measure(MLErrorMeasure):
    """R2 measure for machine learning."""

    SMALLER_IS_BETTER = False

    def _compute_measure(self, outputs, predictions, multioutput=True):
        """Compute R2.

        :param ndarray outputs: reference outputs.
        :param ndarray predictions: predicted outputs.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: R2 value.
        :rtype: float or ndarray(float)
        """
        multioutput = "raw_values" if multioutput else "uniform_average"
        return r2_score(outputs, predictions, multioutput=multioutput)

    def evaluate_kfolds(self, n_folds=5, samples=None, multioutput=True):
        """Evaluate quality measure using the k-folds technique.

        :param int n_folds: number of folds. Default: 5.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        samples = self._assure_samples(samples)
        inds = samples
        folds = array_split(inds, n_folds)

        in_grp = self.algo.learning_set.INPUT_GROUP
        out_grp = self.algo.learning_set.OUTPUT_GROUP
        inputs = self.algo.learning_set.get_data_by_group(in_grp)
        outputs = self.algo.learning_set.get_data_by_group(out_grp)

        multiout = "raw_values" if multioutput else "uniform_average"

        num = 0
        ymean = mean(outputs, axis=0)
        ymean = atleast_2d(ymean)
        ymean = repeat(ymean, outputs.shape[0], axis=0)
        den = mean_squared_error(outputs, ymean, multioutput=multiout) * len(ymean)
        for n_fold in range(n_folds):
            fold = folds[n_fold]
            train = npdelete(inds, fold)
            self.algo.learn(samples=train)
            expected = outputs[fold]
            predicted = self.algo.predict(inputs[fold])
            tmp = mean_squared_error(expected, predicted, multioutput=multiout)
            num += tmp * len(fold)

        quality = 1 - num / den

        return quality

    def evaluate_bootstrap(self, n_replicates=100, samples=None, multioutput=True):
        """Evaluate quality measure using the bootstrap technique.

        :param int n_replicates: number of bootstrap replicates. Default: 100.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        raise NotImplementedError
