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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Root mean squared error measure
===============================

The :mod:`~gemseo.mlearning.qual_measure.mse_measure` module
implements the concept of root mean squared error measures
for machine learning algorithms.

This concept is implemented through the
:class:`.RMSEMeasure` class and
overloads the :meth:`!MSEMeasure.evaluate_*` methods.

The root mean squared error (RMSE) is defined by

.. math::

    \\operatorname{RMSE}(\\hat{y})=\\sqrt{\\frac{1}{n}\\sum_{i=1}^n(\\hat{y}_i-y_i)^2},

where
:math:`\\hat{y}` are the predictions and
:math:`y` are the data points.
"""
from __future__ import absolute_import, division, unicode_literals

from gemseo.mlearning.qual_measure.mse_measure import MSEMeasure


class RMSEMeasure(MSEMeasure):
    """Root mean Squared Error measure for machine learning."""

    def evaluate_learn(self, samples=None, multioutput=True):
        """Evaluate quality measure using the learning dataset.

        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        mse = super(RMSEMeasure, self).evaluate_learn(samples, multioutput)
        return mse ** 0.5

    def evaluate_test(self, test_data, samples=None, multioutput=True):
        """Evaluate quality measure using a test dataset.

        :param Dataset test_data: test data.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        mse = super(RMSEMeasure, self).evaluate_test(test_data, samples, multioutput)
        return mse ** 0.5

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
        mse = super(RMSEMeasure, self).evaluate_kfolds(n_folds, samples, multioutput)
        return mse ** 0.5

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
        mse = super(RMSEMeasure, self).evaluate_bootstrap(
            n_replicates, samples, multioutput
        )
        return mse ** 0.5
