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
F1 error measure
================

The F1 is defined by

.. math::

    F_1 = 2\\frac{\\mathit{precision}\\mathit{recall}}
        {\\mathit{precision}+\\mathit{recall}}

where
:math:`\\mathit{precision}` is the number of correctly predicted positives divided by
the total number of predicted positives and
:math:`\\mathit{recall}` is the number of correctly predicted positives divided by the
total number of true positives.
"""
from __future__ import absolute_import, division, unicode_literals

from sklearn.metrics import f1_score

from gemseo.mlearning.qual_measure.error_measure import MLErrorMeasure


class F1Measure(MLErrorMeasure):
    """F1 measure for machine learning."""

    SMALLER_IS_BETTER = False

    def _compute_measure(self, outputs, predictions, multioutput=False, **options):
        """Compute MSE.

        :param ndarray outputs: reference outputs.
        :param ndarray predictions: predicted outputs.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: MSE value.
        :rtype: float or ndarray(float)
        """
        if multioutput:
            raise NotImplementedError("F1 is only defined for single target.")
        return f1_score(outputs, predictions, average="weighted")
