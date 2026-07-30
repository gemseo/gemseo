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
"""The base class for classification models."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy import arange
from numpy import unique
from numpy import zeros

from gemseo.machine_learning.classification.core.base_classifier_settings import (
    BaseClassifierSettings,
)
from gemseo.machine_learning.core.model.base_supervised import BaseMLSupervisedModel
from gemseo.machine_learning.core.model.base_supervised import (
    SavedObjectType as MLSupervisedModelSavedObjectType,
)
from gemseo.util.typing import NumberArray

if TYPE_CHECKING:
    from gemseo.machine_learning.core.model.base_ml_model import DataType
    from gemseo.util.typing import RealArray

SavedObjectType = (
    MLSupervisedModelSavedObjectType | Sequence[str] | dict[str, NumberArray] | int
)


class BaseClassifier(BaseMLSupervisedModel):
    """The base class for classification models."""

    n_classes: int
    """The number of classes."""

    settings_class = BaseClassifierSettings

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

    @BaseMLSupervisedModel.DataFormatters.format_input_output()
    def predict_proba(
        self,
        input_data: DataType,
        hard: bool = True,
    ) -> DataType:
        """Predict the probability of belonging to each class from input data.

        The user can specify these input data either as a numpy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{'a': array([1.]), 'b': array([2., 3.])}`.

        If the numpy arrays are of dimension 2,
        their i-th rows represent the input data of the i-th sample;
        while if the numpy arrays are of dimension 1,
        there is a single sample.

        The type of the output data and the dimension of the output arrays
        will be consistent
        with the type of the input data and the size of the input arrays.

        Args:
            input_data: The input data.
            hard: Whether classification should be hard (True) or soft (False).

        Returns:
            The probability of belonging to each class.
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
        sample_indices = arange(n_samples)[:, None]
        output_indices = arange(n_outputs)[None, :]
        probas[sample_indices, prediction, output_indices] = 1
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
