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
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Scaling a variable with a linear transformation.

The :class:`.Scaler` class implements the default scaling method
applying to some parameter :math:`z`:

.. math::

    \\bar{z} := \\text{offset} + \\text{coefficient}\\times z

where :math:`\\bar{z}` is the scaled version of :math:`z`.
This scaling method is a linear transformation
parameterized by an offset and a coefficient.

In this default scaling method,
the offset is equal to 0 and the coefficient is equal to 1.
Consequently,
the scaling operation is the identity: :math:`\\bar{z}=z`.
This method has to be overloaded.

.. seealso::

   :mod:`~gemseo.mlearning.transformers.scaler.min_max_scaler`
   :mod:`~gemseo.mlearning.transformers.scaler.standard_scaler`
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Final

from numpy import atleast_1d
from numpy import diag
from numpy import full
from numpy import tile

from gemseo.mlearning.transformers.base_transformer import BaseTransformer
from gemseo.mlearning.transformers.base_transformer import TransformerFitOptionType

if TYPE_CHECKING:
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)


class Scaler(BaseTransformer):
    """Data scaler."""

    __OFFSET: Final[str] = "offset"
    __COEFFICIENT: Final[str] = "coefficient"

    def __init__(
        self,
        name: str = "",
        offset: float | RealArray = 0.0,
        coefficient: float | RealArray = 1.0,
    ) -> None:
        """
        Args:
            name: A name for this transformer.
            offset: The offset of the linear transformation.
            coefficient: The coefficient of the linear transformation.
        """  # noqa: D205 D212
        super().__init__(name)
        self.offset = offset
        self.coefficient = coefficient

    @property
    def offset(self) -> RealArray:
        """The scaling offset."""
        return self.parameters[self.__OFFSET]

    @property
    def coefficient(self) -> RealArray:
        """The scaling coefficient."""
        return self.parameters[self.__COEFFICIENT]

    @offset.setter
    def offset(self, value: float | RealArray) -> None:
        self.parameters[self.__OFFSET] = atleast_1d(value)

    @coefficient.setter
    def coefficient(self, value: float | RealArray) -> None:
        self.parameters[self.__COEFFICIENT] = atleast_1d(value)

    def _fit(self, data: RealArray, *args: TransformerFitOptionType) -> None:
        if self.parameters[self.__COEFFICIENT].size == 1:
            self.parameters[self.__COEFFICIENT] = full(
                data.shape[-1], self.parameters[self.__COEFFICIENT][0]
            )

        if self.parameters[self.__OFFSET].size == 1:
            self.parameters[self.__OFFSET] = full(
                data.shape[-1], self.parameters[self.__OFFSET][0]
            )
        LOGGER.warning(
            (
                "The %s.fit() function does nothing; "
                "the instance of %s uses the coefficient and offset "
                "passed at its initialization"
            ),
            self.__class__.__name__,
            self.__class__.__name__,
        )

    @BaseTransformer._use_2d_array
    def transform(self, data: RealArray) -> RealArray:  # noqa: D102
        return data @ diag(self.coefficient) + self.offset

    @BaseTransformer._use_2d_array
    def inverse_transform(self, data: RealArray) -> RealArray:  # noqa: D102
        return (data - self.offset) @ diag(1 / self.coefficient)

    @BaseTransformer._use_2d_array
    def compute_jacobian(self, data: RealArray) -> RealArray:  # noqa: D102
        return tile(diag(self.coefficient), (len(data), 1, 1))

    @BaseTransformer._use_2d_array
    def compute_jacobian_inverse(self, data: RealArray) -> RealArray:  # noqa: D102
        return tile(diag(1 / self.coefficient), (len(data), 1, 1))
