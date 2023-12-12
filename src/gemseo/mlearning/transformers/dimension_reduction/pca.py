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

from typing import TYPE_CHECKING

from numpy import ndarray
from numpy import sqrt
from numpy import tile
from sklearn.decomposition import PCA as SKLPCA

from gemseo.mlearning.transformers.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.mlearning.transformers.scaler.scaler import Scaler
from gemseo.mlearning.transformers.scaler.standard_scaler import StandardScaler

if TYPE_CHECKING:
    from gemseo.mlearning.transformers.transformer import TransformerFitOptionType


class PCA(DimensionReduction):
    """Principal component dimension reduction algorithm."""

    def __init__(
        self,
        name: str = "",
        n_components: int | None = None,
        scale: bool = False,
        **parameters: float | int | str | bool | None,
    ) -> None:
        """
        Args:
            scale: Whether to scale the data before applying the PCA.
            **parameters: The optional parameters for sklearn PCA constructor.
        """  # noqa: D205 D212
        super().__init__(name, n_components=n_components, **parameters)
        self.algo = SKLPCA(n_components, **parameters)
        self.__scaler = StandardScaler() if scale else Scaler()
        self.__data_is_scaled = scale

    def _fit(self, data: ndarray, *args: TransformerFitOptionType) -> None:
        self.algo.fit(self.__scaler.fit_transform(data))
        self.parameters["n_components"] = self.algo.n_components_

    @DimensionReduction._use_2d_array
    def transform(self, data: ndarray) -> ndarray:  # noqa: D102
        return self.algo.transform(self.__scaler.transform(data))

    @DimensionReduction._use_2d_array
    def inverse_transform(self, data: ndarray) -> ndarray:  # noqa: D102
        return self.__scaler.inverse_transform(self.algo.inverse_transform(data))

    @DimensionReduction._use_2d_array
    def compute_jacobian(self, data: ndarray) -> ndarray:  # noqa: D102
        return tile(
            self.algo.components_, (len(data), 1, 1)
        ) @ self.__scaler.compute_jacobian(data)

    @DimensionReduction._use_2d_array
    def compute_jacobian_inverse(self, data: ndarray) -> ndarray:  # noqa: D102
        _data = self.algo.inverse_transform(data)
        return self.__scaler.compute_jacobian_inverse(_data) @ tile(
            self.algo.components_.T, (len(data), 1, 1)
        )

    @property
    def components(self) -> ndarray:
        """The principal components."""
        return sqrt(self.algo.singular_values_) * self.algo.components_.T

    @property
    def data_is_scaled(self) -> bool:
        """Whether the transformer scales the data before reducing its dimension."""
        return self.__data_is_scaled
