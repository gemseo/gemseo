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
"""This module contains the base class for the unsupervised machine learning algorithms.

The :mod:`~gemseo.mlearning.core.unsupervised` module implements
the concept of unsupervised machine learning models,
where the data has no notion of input or output.

This concept is implemented through the :class:`.MLUnsupervisedAlgo` class,
which inherits from the :class:`.MLAlgo` class.
"""
from __future__ import division, unicode_literals

from typing import Iterable, NoReturn, Optional, Sequence

from numpy import hstack, ndarray

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import MLAlgo, MLAlgoParameterType, TransformerType


class MLUnsupervisedAlgo(MLAlgo):
    """Unsupervised machine learning algorithm.

    Inheriting classes shall overload the
    :meth:`!MLUnsupervisedAlgo._fit` method.

    Attributes:
        input_names (List[str]): The names of the variables.
    """

    ABBR = "MLUnupervisedAlgo"

    def __init__(
        self,
        data,  # type: Dataset
        transformer=None,  # type: Optional[TransformerType]
        var_names=None,  # type: Optional[Iterable[str]]
        **parameters  # type: MLAlgoParameterType
    ):  # type: (...) -> None
        """
        Args:
            var_names: The names of the variables.
                If None, consider all variables mentioned in the learning dataset.
        """
        super(MLUnsupervisedAlgo, self).__init__(
            data, transformer=transformer, var_names=var_names, **parameters
        )
        self.var_names = var_names or data.variables

    def _learn(
        self,
        indices,  # type: Optional[Sequence[int]]
    ):  # type: (...) -> None
        if set(self.var_names) == set(self.learning_set.variables):
            data = []
            for group in self.learning_set.groups:
                sub_data = self.learning_set.get_data_by_group(group)
                if group in self.transformer:
                    sub_data = self.transformer[group].fit_transform(sub_data)
                data.append(sub_data)
            data = hstack(data)
        else:
            data = []
            for name in self.var_names:
                sub_data = self.learning_set.get_data_by_names([name], False)
                if name in self.transformer:
                    sub_data = self.transformer[name].fit_transform(sub_data)
                data.append(sub_data)
            data = hstack(data)

        if indices is not None:
            data = data[indices]

        self._fit(data)

    def _fit(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> NoReturn
        """Fit model on data.

        Args:
            data: The data with shape (n_samples, n_variables).
        """
        raise NotImplementedError
