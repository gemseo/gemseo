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

from future import standard_library
from sklearn.metrics import r2_score

from gemseo.mlearning.qual_measure.error_measure import MLErrorMeasure

standard_library.install_aliases()


class R2Measure(MLErrorMeasure):
    """ R2 measure for machine learning. """

    def _compute_measure(self, outputs, predictions, multioutput=True):
        """Compute MSE.

        :param ndarray outputs: reference outputs.
        :param ndarray predictions: predicted outputs.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: MSE value.
        """
        multioutput = "raw_values" if multioutput else "uniform_average"
        return r2_score(outputs, predictions, multioutput=multioutput)
