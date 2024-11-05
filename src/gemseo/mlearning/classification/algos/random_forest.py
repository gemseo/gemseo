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
#        :author: Francois Gallard, Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The random forest algorithm for classification.

The random forest classification model uses averaging methods on an ensemble
of decision trees.

Dependence
----------
The classifier relies on the RandomForestClassifier class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.ensemble.RandomForestClassifier.html>`_.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import ndarray
from numpy import newaxis
from numpy import stack
from sklearn.ensemble import RandomForestClassifier as SKLRandForest

from gemseo.mlearning.classification.algos.base_classifier import BaseClassifier
from gemseo.mlearning.classification.algos.random_forest_settings import (
    RandomForestClassifier_Settings,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class RandomForestClassifier(BaseClassifier):
    """The random forest classification algorithm."""

    SHORT_ALGO_NAME: ClassVar[str] = "RF"
    LIBRARY: ClassVar[str] = "scikit-learn"

    Settings: ClassVar[type[RandomForestClassifier_Settings]] = (
        RandomForestClassifier_Settings
    )

    def _post_init(self):
        super()._post_init()
        self.algo = SKLRandForest(
            n_estimators=self._settings.n_estimators,
            random_state=self._settings.random_state,
            **self._settings.parameters,
        )

    def _fit(
        self,
        input_data: RealArray,
        output_data: ndarray,
    ) -> None:
        if output_data.shape[1] == 1:
            output_data = output_data.ravel()
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data: RealArray,
    ) -> ndarray:
        return self.algo.predict(input_data).astype(int).reshape((len(input_data), -1))

    def _predict_proba_soft(
        self,
        input_data: RealArray,
    ) -> RealArray:
        probas = self.algo.predict_proba(input_data)
        if probas[0].ndim == 1:
            return probas[..., newaxis]

        return stack(probas, axis=-1)
