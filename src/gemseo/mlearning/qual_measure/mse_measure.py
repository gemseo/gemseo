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
Mean squared error measure
==========================

The :mod:`~gemseo.mlearning.qual_measure.mse_measure` module
implements the concept of means squared error measures
for machine learning algorithms.

This concept is implemented through the
:class:`.MSEMeasure` class and
overloads the :meth:`!MLErrorMeasure._compute_measure` method.

The mean squared error (MSE) is defined by

.. math::

    \\operatorname{MSE}(\\hat{y})=\\frac{1}{n}\\sum_{i=1}^n(\\hat{y}_i-y_i)^2,

where
:math:`\\hat{y}` are the predictions and
:math:`y` are the data points.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from sklearn.metrics import mean_squared_error

from gemseo.mlearning.qual_measure.error_measure import MLErrorMeasure

standard_library.install_aliases()


class MSEMeasure(MLErrorMeasure):
    """ Mean Squared Error measure for machine learning. """

    def _compute_measure(self, outputs, predictions, multioutput=True):
        """Compute MSE.

        :param ndarray outputs: reference outputs.
        :param ndarray predictions: predicted outputs.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: MSE value.
        """
        multioutput = "raw_values" if multioutput else "uniform_average"
        return mean_squared_error(outputs, predictions, multioutput=multioutput)
