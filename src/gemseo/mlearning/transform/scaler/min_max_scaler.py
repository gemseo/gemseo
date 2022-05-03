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
"""Scaling a variable with a geometrical linear transformation.

The :class:`.MinMaxScaler` class implements the MinMax scaling method
applying to some parameter :math:`z`:

.. math::

    \\bar{z} := \\text{offset} + \\text{coefficient}\\times z
    = \\frac{z-\\text{min}(z)}{(\\text{max}(z)-\\text{min}(z))},

where :math:`\\text{offset}=-\\text{min}(z)/(\\text{max}(z)-\\text{min}(z))`
and :math:`\\text{coefficient}=1/(\\text{max}(z)-\\text{min}(z))`.

In the MinMax scaling method,
the scaling operation linearly transforms the original variable :math:`z`
such that the minimum of the original data corresponds to 0 and the maximum to 1.
"""
from __future__ import annotations

from numpy import ndarray

from gemseo.mlearning.transform.scaler.scaler import Scaler
from gemseo.mlearning.transform.transformer import TransformerFitOptionType


class MinMaxScaler(Scaler):
    """Min-max scaler."""

    def __init__(
        self,
        name: str = "MinMaxScaler",
        offset: float = 0.0,
        coefficient: float = 1.0,
    ) -> None:
        """
        Args:
            name: A name for this transformer.
            offset: The offset of the linear transformation.
            coefficient: The coefficient of the linear transformation.
        """
        super().__init__(name, offset, coefficient)

    def _fit(
        self,
        data: ndarray,
        *args: TransformerFitOptionType,
    ) -> None:
        l_b = data.min(0)
        u_b = data.max(0)
        self.offset = -l_b / (u_b - l_b)
        self.coefficient = 1 / (u_b - l_b)
