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
"""This module contains the base class for classification algorithms.

The :mod:`~gemseo.mlearning.classification.classification` module
implements classification algorithms,
whose goal is to assess the membership of input data to classes.

Classification algorithms provide methods for predicting classes of new input data,
as well as predicting the probabilities of belonging to each of the classes
wherever possible.

This concept is implemented through the :class:`.MLClassificationAlgo` class
which inherits from the :class:`.MLSupervisedAlgo` class.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Union

from numpy import ndarray
from numpy import unique
from numpy import zeros

from gemseo.mlearning.core.supervised import MLSupervisedAlgo
from gemseo.mlearning.core.supervised import (
    SavedObjectType as MLSupervisedAlgoSavedObjectType,
)

if TYPE_CHECKING:
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.ml_algo import DataType
    from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
    from gemseo.mlearning.core.ml_algo import TransformerType

SavedObjectType = Union[
    MLSupervisedAlgoSavedObjectType, Sequence[str], dict[str, ndarray], int
]


class MLClassificationAlgo(MLSupervisedAlgo):
    """Classification Algorithm.

    Inheriting classes shall implement the :meth:`!MLSupervisedAlgo._fit` and
    :meth:`!MLClassificationAlgo._predict` methods, and
    :meth:`!MLClassificationAlgo._predict_proba_soft` method if possible.
    """

    n_classes: int
    """The number of classes."""

    def __init__(  # noqa: D107
        self,
        data: IODataset,
        transformer: TransformerType = MLSupervisedAlgo.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        **parameters: MLAlgoParameterType,
    ) -> None:
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            **parameters,
        )
        self.n_classes = None

    def _learn(
        self,
        indices: Sequence[int] | None,
        fit_transformers: bool,
    ) -> None:
        output_data = self.learning_set.get_view(
            group_names=self.learning_set.OUTPUT_GROUP,
            variable_names=self.output_names,
        ).to_numpy()
        self.n_classes = unique(output_data).shape[0]
        super()._learn(indices, fit_transformers=fit_transformers)

    @MLSupervisedAlgo.DataFormatters.format_input_output
    def predict_proba(
        self,
        input_data: DataType,
        hard: bool = True,
    ) -> ndarray:
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
        input_data: ndarray,
        hard: bool = True,
    ) -> ndarray:
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
        input_data: ndarray,
    ) -> ndarray:
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
        input_data: ndarray,
    ) -> ndarray:
        """Predict the probability of belonging to each class.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The probability of belonging to each class
                with shape (n_samples, n_classes).
        """

    def _get_objects_to_save(self) -> SavedObjectType:
        objects = super()._get_objects_to_save()
        objects["n_classes"] = self.n_classes
        return objects
