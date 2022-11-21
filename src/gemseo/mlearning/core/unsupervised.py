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
"""This module contains the base class for the unsupervised machine learning algorithms.

The :mod:`~gemseo.mlearning.core.unsupervised` module implements
the concept of unsupervised machine learning models,
where the data has no notion of input or output.

This concept is implemented through the :class:`.MLUnsupervisedAlgo` class,
which inherits from the :class:`.MLAlgo` class.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar
from typing import Iterable
from typing import NoReturn
from typing import Sequence

from numpy import hstack
from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.ml_algo import TransformerType


class MLUnsupervisedAlgo(MLAlgo):
    """Unsupervised machine learning algorithm.

    Inheriting classes shall overload the
    :meth:`!MLUnsupervisedAlgo._fit` method.
    """

    input_names: list[str]
    """The names of the variables."""

    SHORT_ALGO_NAME: ClassVar[str] = "MLUnsupervisedAlgo"

    def __init__(
        self,
        data: Dataset,
        transformer: TransformerType = MLAlgo.IDENTITY,
        var_names: Iterable[str] | None = None,
        **parameters: MLAlgoParameterType,
    ) -> None:
        """
        Args:
            var_names: The names of the variables.
                If ``None``, consider all variables mentioned in the learning dataset.
        """
        super().__init__(
            data, transformer=transformer, var_names=var_names, **parameters
        )
        self.var_names = var_names or data.variables

    def _learn(
        self,
        indices: Sequence[int] | None,
        fit_transformers: bool,
    ) -> None:
        if set(self.var_names) == set(self.learning_set.variables):
            data = []
            for group in self.learning_set.groups:
                sub_data = self.learning_set.get_data_by_group(group)
                if fit_transformers and group in self.transformer:
                    sub_data = self.transformer[group].fit_transform(sub_data)
                data.append(sub_data)
            data = hstack(data)
        else:
            data = []
            for name in self.var_names:
                sub_data = self.learning_set.get_data_by_names([name], False)
                if fit_transformers and name in self.transformer:
                    sub_data = self.transformer[name].fit_transform(sub_data)
                data.append(sub_data)
            data = hstack(data)

        if indices is not None:
            data = data[indices]

        self._fit(data)

    @abstractmethod
    def _fit(
        self,
        data: ndarray,
    ) -> NoReturn:
        """Fit model on data.

        Args:
            data: The data with shape (n_samples, n_variables).
        """
