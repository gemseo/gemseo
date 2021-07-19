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
"""The Partial Least Square (PLS) regression to reduce the dimension of a variable.

The :class:`PLS` class wraps the PCA from Scikit-learn.

Dependence
----------
This dimension reduction algorithm relies on the PLSRegression class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.cross_decomposition.PLSRegression.html>`_.
"""
from __future__ import division, unicode_literals

from typing import NoReturn, Union

from numpy import matmul, ndarray
from sklearn.cross_decomposition import PLSRegression

from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)


class PLS(DimensionReduction):
    """Partial Least Square regression."""

    CROSSED = True

    def __init__(
        self,
        name="PLS",  # type: str
        n_components=5,  # type: int
        **parameters  # type: Union[float,int,bool]
    ):  # type: (...) -> None
        """
        Args:
            **parameters: The optional parameters for sklearn PCA constructor.
        """
        super(PLS, self).__init__(name, n_components=n_components, **parameters)
        self.algo = PLSRegression(n_components, **parameters)

    def fit(
        self,
        data,  # type: ndarray
        other_data,  # type: ndarray
    ):  # type: (...) -> None
        """Fit the transformer to the data.

        Args:
            The data to be fitted.
        """
        self.algo.fit(data, other_data)

    def transform(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        return self.algo.transform(data)

    def inverse_transform(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        inv_data = matmul(data, self.algo.x_loadings_.T)
        inv_data *= self.algo.x_std_
        inv_data += self.algo.x_mean_
        return inv_data

    def compute_jacobian(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> NoReturn
        raise NotImplementedError

    def compute_jacobian_inverse(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> NoReturn
        raise NotImplementedError

    @property
    def components(self):  # type: (...) -> NoReturn
        """The principal components."""
        raise NotImplementedError
