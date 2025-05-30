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
r"""The F1 score to assess the quality of a classifier.

The F1 score is defined by

.. math::

    F_1 = 2\frac{\mathit{precision}\times\mathit{recall}}
        {\mathit{precision}+\mathit{recall}}

where
:math:`\mathit{precision}` is the number of correctly predicted positives
divided by the total number of *predicted* positives
and :math:`\mathit{recall}` is the number of correctly predicted positives
divided by the total number of *true* positives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import f1_score

from gemseo.mlearning.classification.quality.base_classifier_quality import (
    BaseClassifierQuality,
)

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.mlearning.classification.algos.base_classifier import BaseClassifier
    from gemseo.mlearning.core.quality.base_ml_algo_quality import MeasureType


class F1Measure(BaseClassifierQuality):
    """The F1 score to assess the quality of a classifier."""

    SMALLER_IS_BETTER = False

    algo: BaseClassifier

    def __init__(
        self,
        algo: BaseClassifier,
        fit_transformers: bool = BaseClassifierQuality._FIT_TRANSFORMERS,
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
            msg = "F1 is only defined for single target."
            raise NotImplementedError(msg)
        return f1_score(outputs, predictions, average="weighted")
