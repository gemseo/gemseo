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
"""The Support Vector Machine algorithm for classification.

This module implements the SVMClassifier class.
A support vector machine (SVM) passes the data through a kernel
in order to increase its dimension
and thereby make the classes linearly separable.

Dependence
----------
The classifier relies on the SVC class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.svm.SVC.html>`_.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from sklearn.svm import SVC

from gemseo.mlearning.classification.algos.base_classifier import BaseClassifier
from gemseo.mlearning.classification.algos.svm_settings import SVMClassifier_Settings

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.typing import RealArray


class SVMClassifier(BaseClassifier):
    """The Support Vector Machine algorithm for classification."""

    SHORT_ALGO_NAME: ClassVar[str] = "SVM"
    LIBRARY: ClassVar[str] = "scikit-learn"

    Settings: ClassVar[type[SVMClassifier_Settings]] = SVMClassifier_Settings

    def _post_init(self):
        super()._post_init()
        self.algo = SVC(
            C=self._settings.C,
            kernel=self._settings.kernel,
            probability=self._settings.probability,
            random_state=self._settings.random_state,
            **self._settings.parameters,
        )

    def _fit(
        self,
        input_data: RealArray,
        output_data: ndarray,
    ) -> None:
        self.algo.fit(input_data, output_data.ravel())

    def _predict(
        self,
        input_data: RealArray,
    ) -> ndarray:
        return self.algo.predict(input_data).astype(int).reshape((len(input_data), -1))

    def _predict_proba_soft(
        self,
        input_data: RealArray,
    ) -> RealArray:
        if not self._settings.probability:
            msg = (
                "SVMClassifier soft probability prediction is only available if the "
                "parameter 'probability' is set to True."
            )
            raise NotImplementedError(msg)
        return self.algo.predict_proba(input_data)[..., None]
