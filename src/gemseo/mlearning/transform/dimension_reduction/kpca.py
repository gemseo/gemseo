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
"""The Kernel Principal Component Analysis (KPCA) to reduce the dimension of a variable.

The :class:`.KPCA` class implements the KCPA wraps the KPCA from Scikit-learn.

Dependence
----------
This dimension reduction algorithm relies on the PCA class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.decomposition.PCA.html>`_.
"""
from __future__ import annotations

from numpy import ndarray
from sklearn.decomposition import KernelPCA

from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.mlearning.transform.transformer import TransformerFitOptionType


class KPCA(DimensionReduction):
    """Kernel principal component dimension reduction algorithm."""

    def __init__(
        self,
        name: str = "KPCA",
        n_components: int | None = None,
        fit_inverse_transform: bool = True,
        kernel: str = "linear",
        **parameters: float | int | str | None,
    ) -> None:
        """
        Args:
            fit_inverse_transform: If True, learn the inverse transform
                for non-precomputed kernels.
            kernel: The name of the kernel,
                either 'linear', 'poly', 'rbf', 'sigmoid', 'cosine' or 'precomputed'.
            **parameters: The optional parameters for sklearn KPCA constructor.
        """
        super().__init__(name, n_components=n_components, **parameters)
        self.algo = KernelPCA(
            n_components,
            fit_inverse_transform=fit_inverse_transform,
            kernel=kernel,
            **parameters,
        )

    def _fit(self, data: ndarray, *args: TransformerFitOptionType) -> None:
        self.algo.fit(data)
        self.parameters["n_components"] = len(self.algo.eigenvalues_)

    @DimensionReduction._use_2d_array
    def transform(self, data: ndarray) -> ndarray:
        return self.algo.transform(data)

    @DimensionReduction._use_2d_array
    def inverse_transform(self, data: ndarray) -> ndarray:
        return self.algo.inverse_transform(data)
