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
"""The Principal Component Analysis (PCA) to reduce the dimension of a variable.

The :class:`PCA` class wraps the PCA from Scikit-learn.

Dependence
----------
This dimension reduction algorithm relies on the PCA class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.decomposition.PCA.html>`_.
"""
from __future__ import division, unicode_literals

from typing import Optional, Union

from numpy import ndarray, sqrt
from sklearn.decomposition import PCA as SKLPCA

from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.mlearning.transform.transformer import TransformerFitOptionType


class PCA(DimensionReduction):
    """Principal component dimension reduction algorithm."""

    def __init__(
        self,
        name="PCA",  # type: str,
        n_components=5,  # type: int
        **parameters  # type: Optional[Union[float,int,str,bool]]
    ):  # type: (...) -> None
        """
        Args:
            **parameters: The optional parameters for sklearn PCA constructor.
        """
        super(PCA, self).__init__(name, n_components=n_components, **parameters)
        self.algo = SKLPCA(n_components, **parameters)

    def fit(
        self,
        data,  # type: ndarray
        *args  # type: TransformerFitOptionType
    ):  # type: (...) -> None
        self.algo.fit(data)

    def transform(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        return self.algo.transform(data)

    def inverse_transform(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        return self.algo.inverse_transform(data)

    def compute_jacobian(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        return self.algo.components_

    def compute_jacobian_inverse(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        return self.algo.components_.T

    @property
    def components(self):  # type: (...) -> ndarray
        """The principal components."""
        return sqrt(self.algo.singular_values_) * self.algo.components_.T
