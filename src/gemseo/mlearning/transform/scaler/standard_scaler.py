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
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Scaling a variable with a statistical linear transformation.

The :class:`.StandardScaler` class implements the Standard scaling method
applying to some parameter :math:`z`:

.. math::

    \\bar{z} := \\text{offset} + \\text{coefficient}\\times z
    = \\frac{z-\\text{mean}(z)}{\\text{std}(z)}

where :math:`\\text{offset}=-\\text{mean}(z)/\\text{std}(z)` and
:math:`\\text{coefficient}=1/\\text{std}(z)`.

In this standard scaling method,
the scaling operation linearly transforms the original variable math:`z`
such that in the scaled space,
the original data have zero mean and unit standard deviation.
"""
from __future__ import division, unicode_literals

from numpy import mean, ndarray, std
from past.utils import old_div

from gemseo.mlearning.transform.scaler.scaler import Scaler
from gemseo.mlearning.transform.transformer import TransformerFitOptionType


class StandardScaler(Scaler):
    """Standard scaler."""

    def __init__(
        self,
        name="StandardScaler",  # type: str
        offset=0.0,  # type: float
        coefficient=1.0,  # type: float
    ):  # type: (...) -> None
        """
        Args:
            name: A name for this transformer.
            offset: The offset of the linear transformation.
            coefficient: The coefficient of the linear transformation.
        """
        super(StandardScaler, self).__init__(name, offset, coefficient)

    def fit(
        self,
        data,  # type: ndarray
        *args  # type: TransformerFitOptionType
    ):  # type: (...) -> None
        average = mean(data, 0)
        std_ = std(data, 0)
        self.offset = old_div(-average, std_)
        self.coefficient = 1.0 / std_
