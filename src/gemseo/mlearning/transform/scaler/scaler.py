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
"""Scaling a variable with a linear transformation.

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

   :mod:`~gemseo.mlearning.transform.scaler.min_max_scaler`
   :mod:`~gemseo.mlearning.transform.scaler.standard_scaler`
"""
from __future__ import annotations

import logging

from numpy import atleast_1d
from numpy import diag
from numpy import full
from numpy import ndarray
from numpy import tile

from gemseo.mlearning.transform.transformer import Transformer
from gemseo.mlearning.transform.transformer import TransformerFitOptionType
from gemseo.utils.python_compatibility import Final

LOGGER = logging.getLogger(__name__)


class Scaler(Transformer):
    """Data scaler."""

    __OFFSET: Final[str] = "offset"
    __COEFFICIENT: Final[str] = "coefficient"

    def __init__(
        self,
        name: str = "Scaler",
        offset: float | ndarray = 0.0,
        coefficient: float | ndarray = 1.0,
    ) -> None:
        """
        Args:
            name: A name for this transformer.
            offset: The offset of the linear transformation.
            coefficient: The coefficient of the linear transformation.
        """
        super().__init__(name)
        self.offset = offset
        self.coefficient = coefficient

    @property
    def offset(self) -> ndarray:
        """The scaling offset."""
        return self.parameters[self.__OFFSET]

    @property
    def coefficient(self) -> ndarray:
        """The scaling coefficient."""
        return self.parameters[self.__COEFFICIENT]

    @offset.setter
    def offset(self, value: float | ndarray) -> None:
        self.parameters[self.__OFFSET] = atleast_1d(value)

    @coefficient.setter
    def coefficient(self, value: float | ndarray) -> None:
        self.parameters[self.__COEFFICIENT] = atleast_1d(value)

    def _fit(self, data: ndarray, *args: TransformerFitOptionType) -> None:
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

    @Transformer._use_2d_array
    def transform(self, data: ndarray) -> ndarray:
        return data @ diag(self.coefficient) + self.offset

    @Transformer._use_2d_array
    def inverse_transform(self, data: ndarray) -> ndarray:
        return (data - self.offset) @ diag(1 / self.coefficient)

    @Transformer._use_2d_array
    def compute_jacobian(self, data: ndarray) -> ndarray:
        return tile(diag(self.coefficient), (len(data), 1, 1))

    @Transformer._use_2d_array
    def compute_jacobian_inverse(self, data: ndarray) -> ndarray:
        return tile(diag(1 / self.coefficient), (len(data), 1, 1))
