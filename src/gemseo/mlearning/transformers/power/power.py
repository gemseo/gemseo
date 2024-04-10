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
#        :author: Matthias De Lozzo, Gilberto Ruiz Jimenez
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A power transform, either Yeo-Johnson or Box-Cox.

Dependence
----------
This transformation algorithm relies on the ``PowerTransformer`` class
of `scikit-learn <https://scikit-learn.org/
stable/modules/generated/
sklearn.preprocessing.PowerTransformer.html>`_.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from sklearn.preprocessing import PowerTransformer

from gemseo.mlearning.transformers.base_transformer import BaseTransformer
from gemseo.mlearning.transformers.base_transformer import TransformerFitOptionType

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Power(BaseTransformer):
    """A power transformation."""

    lambdas_: RealArray
    """The parameters of the power transformation for the selected features."""

    _TRANSFORMER_NAME: ClassVar[str] = "yeo-johnson"
    """The name of the transformer in scikit-learn."""

    def __init__(self, name: str = "", standardize: bool = True) -> None:
        """
        Args:
            name: A name for this transformer. If ``None``, use the class name.
            standardize: Whether to apply zero-mean, unit-variance
                normalization to the transformed output.
        """  # noqa: D205 D212
        super().__init__(name, standardize=standardize)
        self.__power_transformer = PowerTransformer(
            method=self._TRANSFORMER_NAME,
            standardize=standardize,
        )

    def _fit(self, data: RealArray, *args: TransformerFitOptionType) -> None:
        self.__power_transformer.fit(data)
        self.lambdas_ = self.__power_transformer.lambdas_

    @BaseTransformer._use_2d_array
    def transform(self, data: RealArray) -> RealArray:  # noqa: D102
        return self.__power_transformer.transform(data)

    @BaseTransformer._use_2d_array
    def inverse_transform(self, data: RealArray) -> RealArray:  # noqa: D102
        return self.__power_transformer.inverse_transform(data)
