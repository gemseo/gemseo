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
Quality measure
===============

The :mod:`~gemseo.mlearning.qual_measure.quality_measure` module
implements the concept of quality measures for machine learning algorithms.

This concept is implemented through the :class:`.MLQualityMeasure` class.
"""
from __future__ import absolute_import, division, unicode_literals

from numpy import arange


class MLQualityMeasure(object):
    """Quality measure for machine learning."""

    LEARN = "learn"
    TEST = "test"
    LOO = "loo"
    KFOLDS = "kfolds"
    BOOTSTRAP = "bootstrap"

    SMALLER_IS_BETTER = True  # To be overwritten in inheriting classes

    def __init__(self, algo):
        """Constructor.

        :param MLAlgo algo: machine learning algorithm.
        """
        self.algo = algo

    def evaluate(self, method=LEARN, samples=None, **options):
        """Evaluate quality measure.

        :param str method: method to estimate the quality measure.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param options: options of the estimation method (e.g. 'test_data' for
            the 'test' method, 'n_replicates' for the bootstrap one, ...)
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        if method == self.LEARN:
            evaluation = self.evaluate_learn(samples=samples, **options)
        elif method == self.TEST:
            evaluation = self.evaluate_test(samples=samples, **options)
        elif method == self.LOO:
            evaluation = self.evaluate_loo(samples=samples, **options)
        elif method == self.KFOLDS:
            evaluation = self.evaluate_kfolds(samples=samples, **options)
        elif method == self.BOOTSTRAP:
            evaluation = self.evaluate_bootstrap(samples=samples, **options)
        return evaluation

    def evaluate_learn(self, samples=None, multioutput=True):
        """Evaluate quality measure using the learning dataset.

        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        raise NotImplementedError

    def evaluate_test(self, test_data, samples=None, multioutput=True):
        """Evaluate quality measure using a test dataset.

        :param Dataset test_data: test data.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        raise NotImplementedError

    def evaluate_loo(self, samples=None, multioutput=True):
        """Evaluate quality measure using the leave-one-out technique.

        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        n_samples = self.algo.learning_set.n_samples
        return self.evaluate_kfolds(n_folds=n_samples, multioutput=multioutput)

    def evaluate_kfolds(self, n_folds=5, samples=None, multioutput=True):
        """Evaluate quality measure using the k-folds technique.

        :param int n_folds: number of folds. Default: 5.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        raise NotImplementedError

    def evaluate_bootstrap(self, n_replicates=100, samples=None, multioutput=True):
        """Evaluate quality measure using the bootstrap technique.

        :param int n_replicates: number of bootstrap replicates. Default: 100.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        raise NotImplementedError

    @classmethod
    def is_better(cls, val1, val2):
        """Compare quality between two values and return True if the first one is better
        than the second one.

        For most measures, a smaller value is "better" than a larger one (MSE
        etc.). But for some, like an R2-measure, higher values are better than
        smaller ones. This comparison method correctly handles this,
        regardless of the type of measure.

        :param float val1: first quality measure value.
        :param float val2: second quality measure value.
        :return: Indicator for whether val1 is of better quality than val2.
        :rtype: bool
        """
        if cls.SMALLER_IS_BETTER:
            result = val1 < val2
        else:
            result = val1 > val2
        return result

    def _assure_samples(self, samples):
        """Get list of all samples if samples is None.

        :param list(int) samples: List of samples. Can also be None.
        :return: list of samples.
        :rtype: list(int)
        """
        if samples is None:
            samples = arange(self.algo.learning_set.n_samples)
        return samples
