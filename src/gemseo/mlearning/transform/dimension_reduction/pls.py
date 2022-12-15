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

The :class:`.PLS` class wraps the PCA from Scikit-learn.

Dependence
----------
This dimension reduction algorithm relies on the PLSRegression class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.cross_decomposition.PLSRegression.html>`_.
"""
from __future__ import annotations

from numpy import ndarray
from sklearn.cross_decomposition import PLSRegression

from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.utils.python_compatibility import Final


class PLS(DimensionReduction):
    """Partial Least Square regression."""

    CROSSED: Final[bool] = True

    def __init__(
        self,
        name: str = "PLS",
        n_components: int | None = None,
        **parameters: float | int | bool,
    ) -> None:
        """
        Args:
            **parameters: The optional parameters for sklearn PCA constructor.
        """
        super().__init__(name, n_components=n_components, **parameters)
        self.algo = PLSRegression(n_components, **parameters)

    def _fit(self, data: ndarray, other_data: ndarray) -> None:
        """Fit the transformer to the data.

        Args:
            The data to be fitted.
        """
        if self.algo.n_components is None:
            self.algo.n_components = min(min(data.shape), min(other_data.shape))

        self.algo.fit(data, other_data)
        self.parameters["n_components"] = self.algo.n_components

    def transform(self, data: ndarray) -> ndarray:
        return self.algo.transform(data)

    def inverse_transform(self, data: ndarray) -> ndarray:
        return self.algo.inverse_transform(data)
