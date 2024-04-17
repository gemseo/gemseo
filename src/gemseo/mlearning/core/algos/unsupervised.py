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

The :mod:`~gemseo.mlearning.core.unsupervised` module implements the concept of
unsupervised machine learning models, where the data has no notion of input or output.

This concept is implemented through the :class:`.BaseMLUnsupervisedAlgo` class, which
inherits from the :class:`.BaseMLAlgo` class.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import NoReturn

from numpy import hstack
from numpy import ndarray

from gemseo.mlearning.core.algos.ml_algo import BaseMLAlgo
from gemseo.mlearning.core.algos.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.algos.ml_algo import TransformerType

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.datasets.dataset import Dataset


class BaseMLUnsupervisedAlgo(BaseMLAlgo):
    """Unsupervised machine learning algorithm.

    Inheriting classes shall overload the :meth:`!BaseMLUnsupervisedAlgo._fit` method.
    """

    input_names: list[str]
    """The names of the variables."""

    SHORT_ALGO_NAME: ClassVar[str] = "BaseMLUnsupervisedAlgo"

    def __init__(
        self,
        data: Dataset,
        transformer: TransformerType = BaseMLAlgo.IDENTITY,
        var_names: Iterable[str] | None = None,
        **parameters: MLAlgoParameterType,
    ) -> None:
        """
        Args:
            var_names: The names of the variables.
                If ``None``, consider all variables mentioned in the learning dataset.
        """  # noqa: D205 D212
        super().__init__(
            data, transformer=transformer, var_names=var_names, **parameters
        )
        self.var_names = var_names or data.variable_names

    def _learn(
        self,
        indices: Sequence[int] | None,
        fit_transformers: bool,
    ) -> None:
        if set(self.var_names) == set(self.learning_set.variable_names):
            names = self.learning_set.group_names
            arg_name = "group_names"
        else:
            names = self.var_names
            arg_name = "variable_names"

        data = []
        method_name = "fit_transform" if fit_transformers else "transform"
        for name in names:
            sub_data = self.learning_set.get_view(**{arg_name: name}).to_numpy()
            if name in self.transformer:
                sub_data = getattr(self.transformer[name], method_name)(sub_data)

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
