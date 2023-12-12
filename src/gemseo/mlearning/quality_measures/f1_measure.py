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
r"""The F1 to measure the quality of a classification algorithm.

The F1 is defined by

.. math::

    F_1 = 2\\frac{\\mathit{precision}\\mathit{recall}}
        {\\mathit{precision}+\\mathit{recall}}

where
:math:`\\mathit{precision}` is the number of correctly predicted positives
divided by the total number of *predicted* positives
and :math:`\\mathit{recall}` is the number of correctly predicted positives
divided by the total number of *true* positives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import f1_score

from gemseo.mlearning.quality_measures.error_measure import MLErrorMeasure

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.mlearning.classification.classification import MLClassificationAlgo
    from gemseo.mlearning.quality_measures.quality_measure import MeasureType


class F1Measure(MLErrorMeasure):
    """The F1 measure for machine learning."""

    SMALLER_IS_BETTER = False

    algo: MLClassificationAlgo

    def __init__(
        self,
        algo: MLClassificationAlgo,
        fit_transformers: bool = MLErrorMeasure._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for classification.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers=fit_transformers)

    def _compute_measure(
        self,
        outputs: ndarray,
        predictions: ndarray,
        multioutput: bool = True,
    ) -> MeasureType:
        if multioutput:
            raise NotImplementedError("F1 is only defined for single target.")
        return f1_score(outputs, predictions, average="weighted")
