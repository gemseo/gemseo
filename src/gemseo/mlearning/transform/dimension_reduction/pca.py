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

The :class:`.PCA` class wraps the PCA from Scikit-learn.

Dependence
----------
This dimension reduction algorithm relies on the PCA class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.decomposition.PCA.html>`_.
"""
from __future__ import annotations

from numpy import ndarray
from numpy import sqrt
from numpy import tile
from sklearn.decomposition import PCA as SKLPCA

from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.mlearning.transform.transformer import TransformerFitOptionType


class PCA(DimensionReduction):
    """Principal component dimension reduction algorithm."""

    def __init__(
        self,
        name: str = "PCA",
        n_components: int | None = None,
        **parameters: float | int | str | bool | None,
    ) -> None:
        """
        Args:
            **parameters: The optional parameters for sklearn PCA constructor.
        """
        super().__init__(name, n_components=n_components, **parameters)
        self.algo = SKLPCA(n_components, **parameters)

    def _fit(self, data: ndarray, *args: TransformerFitOptionType) -> None:
        self.algo.fit(data)
        self.parameters["n_components"] = self.algo.n_components_

    @DimensionReduction._use_2d_array
    def transform(self, data: ndarray) -> ndarray:
        return self.algo.transform(data)

    @DimensionReduction._use_2d_array
    def inverse_transform(self, data: ndarray) -> ndarray:
        return self.algo.inverse_transform(data)

    @DimensionReduction._use_2d_array
    def compute_jacobian(self, data: ndarray) -> ndarray:
        return tile(self.algo.components_, (len(data), 1, 1))

    @DimensionReduction._use_2d_array
    def compute_jacobian_inverse(self, data: ndarray) -> ndarray:
        return tile(self.algo.components_.T, (len(data), 1, 1))

    @property
    def components(self) -> ndarray:
        """The principal components."""
        return sqrt(self.algo.singular_values_) * self.algo.components_.T
