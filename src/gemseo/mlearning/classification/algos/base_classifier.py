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
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The base class for classification algorithms."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Union

from numpy import unique
from numpy import zeros

from gemseo.mlearning.classification.algos.base_classifier_settings import (
    BaseClassifierSettings,
)
from gemseo.mlearning.core.algos.supervised import BaseMLSupervisedAlgo
from gemseo.mlearning.core.algos.supervised import (
    SavedObjectType as MLSupervisedAlgoSavedObjectType,
)
from gemseo.typing import NumberArray
from gemseo.typing import RealArray

if TYPE_CHECKING:
    from gemseo.mlearning.core.algos.ml_algo import DataType

SavedObjectType = Union[
    MLSupervisedAlgoSavedObjectType, Sequence[str], dict[str, NumberArray], int
]


class BaseClassifier(BaseMLSupervisedAlgo):
    """The base class for classification algorithms."""

    n_classes: int
    """The number of classes computed when calling :meth:`.learn`."""

    Settings = BaseClassifierSettings

    def _post_init(self):
        super()._post_init()
        self.n_classes = 0

    def _learn(
        self,
        indices: Sequence[int],
        fit_transformers: bool,
    ) -> None:
        output_data = self.learning_set.get_view(
            group_names=self.learning_set.OUTPUT_GROUP,
            variable_names=self.output_names,
        ).to_numpy()
        self.n_classes = unique(output_data).shape[0]
        super()._learn(indices, fit_transformers=fit_transformers)

    @BaseMLSupervisedAlgo.DataFormatters.format_input_output()
    def predict_proba(
        self,
        input_data: DataType,
        hard: bool = True,
    ) -> DataType:
        """Predict the probability of belonging to each cluster from input data.

        The user can specify these input data either as a numpy array,
        e.g. ``array([1., 2., 3.])``
        or as a dictionary,
        e.g.  ``{'a': array([1.]), 'b': array([2., 3.])}``.

        If the numpy arrays are of dimension 2,
        their i-th rows represent the input data of the i-th sample;
        while if the numpy arrays are of dimension 1,
        there is a single sample.

        The type of the output data and the dimension of the output arrays
        will be consistent
        with the type of the input data and the size of the input arrays.

        Args:
            input_data: The input data.
            hard: Whether clustering should be hard (True) or soft (False).

        Returns:
            The probability of belonging to each cluster.
        """
        return self._predict_proba(input_data, hard)

    def _predict_proba(
        self,
        input_data: RealArray,
        hard: bool = True,
    ) -> RealArray:
        """Predict the probability of belonging to each class.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).
            hard: Whether clustering should be hard (True) or soft (False).

        Returns:
            The probability of belonging to each class
                with shape (n_samples, n_classes).
        """
        if hard:
            return self._predict_proba_hard(input_data)

        return self._predict_proba_soft(input_data)

    def _predict_proba_hard(
        self,
        input_data: RealArray,
    ) -> RealArray:
        """Return 1 if the data belongs to a class, 0 otherwise.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The indicator of belonging to each class with shape (n_samples, n_classes).
        """
        n_samples = len(input_data)
        prediction = self._predict(input_data).astype(int)
        n_outputs = prediction.shape[1]
        probas = zeros((n_samples, self.n_classes, n_outputs))
        for sample in range(n_samples):
            for n_output in range(n_outputs):
                probas[sample, prediction[sample, n_output], n_output] = 1
        return probas

    @abstractmethod
    def _predict_proba_soft(
        self,
        input_data: RealArray,
    ) -> RealArray:
        """Predict the probability of belonging to each class.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The probability of belonging to each class
                with shape (n_samples, n_classes).
        """
