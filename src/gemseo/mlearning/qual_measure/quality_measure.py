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

from future import standard_library

standard_library.install_aliases()


class MLQualityMeasure(object):
    """ Quality measure for machine learning. """

    LEARN = "learn"
    TEST = "test"
    LOO = "loo"
    KFOLDS = "kfolds"
    BOOTSTRAP = "bootstrap"

    def __init__(self, algo):
        """Constructor.

        :param MLAlgo algo: machine learning algorithm.
        """
        self.algo = algo

    def evaluate(self, method=LEARN, **options):
        """Evaluate quality measure.

        :param str method: method to estimate the quality measure.
        :param options: options of the estimation method (e.g. 'test_data' for
            the 'test' method, 'n_replicates' for the boostrap one, ...)
        :return: quality measure value.
        """
        if method == self.LEARN:
            evalutation = self.evaluate_learn(**options)
        elif method == self.TEST:
            evalutation = self.evaluate_test(**options)
        elif method == self.LOO:
            evalutation = self.evaluate_loo(**options)
        elif method == self.KFOLDS:
            evalutation = self.evaluate_kfolds(**options)
        elif method == self.BOOTSTRAP:
            evalutation = self.evaluate_bootstrap(**options)
        return evalutation

    def evaluate_learn(self, multioutput=True):
        """Evaluate quality measure using the learning dataset.

        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        """
        raise NotImplementedError

    def evaluate_test(self, test_data, multioutput=True):
        """Evaluate quality measure using a test dataset.

        :param Dataset test_data: test data.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        """
        raise NotImplementedError

    def evaluate_loo(self, multioutput=True):
        """Evaluate quality measure using the leave-one-out technique.

        :param bool multioutput: if True, return the quality measure for each
            component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        """
        n_samples = self.algo.learning_set.n_samples
        return self.evaluate_kfolds(n_folds=n_samples, multioutput=multioutput)

    def evaluate_kfolds(self, n_folds=5, multioutput=True):
        """Evaluate quality measure using the k-folds technique.

        :param int n_folds: number of folds. Default: 5.
        :param bool multioutput: if True, return the quality measure for each
            component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        """
        raise NotImplementedError

    def evaluate_bootstrap(self, n_replicates=100, multioutput=True):
        """Evaluate quality measure using the bootstrap technique.

        :param int n_replicates: number of bootstrap replicates. Default: 100.
        :param bool multioutput: if True, return the quality measure for each
            component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        """
        raise NotImplementedError
