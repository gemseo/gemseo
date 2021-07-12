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
from __future__ import division, unicode_literals

import logging

from numpy import diag, eye, ndarray

from gemseo.mlearning.transform.transformer import Transformer, TransformerFitOptionType

LOGGER = logging.getLogger(__name__)


class Scaler(Transformer):
    """Data scaler."""

    def __init__(
        self,
        name="Scaler",  # type: str
        offset=0.0,  # type: float
        coefficient=1.0,  # type: float
    ):  # type: (...) -> None
        """
        Args:
            name: A name for this transformer.
            offset: The offset of the linear transformation.
            coefficient: The coefficient of the linear transformation.
        """
        super(Scaler, self).__init__(name, offset=offset, coefficient=coefficient)

    @property
    def offset(self):  # type: (...) -> float
        """The scaling offset."""
        return self.parameters["offset"]

    @property
    def coefficient(self):  # type: (...) -> float
        """The scaling coefficient."""
        return self.parameters["coefficient"]

    @offset.setter
    def offset(
        self,
        value,  # type: float
    ):  # type: (...) -> None
        self.parameters["offset"] = value

    @coefficient.setter
    def coefficient(
        self,
        value,  # type: float
    ):  # type: (...) -> None
        self.parameters["coefficient"] = value

    def fit(
        self,
        data,  # type: ndarray
        *args  # type: TransformerFitOptionType
    ):  # type: (...) -> None
        LOGGER.warning(
            (
                "The %s.fit() function does nothing; "
                "the instance of %s uses the coefficient and offset "
                "passed at its initialization"
            ),
            self.__class__.__name__,
            self.__class__.__name__,
        )

    def transform(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        return self.offset + self.coefficient * data

    def inverse_transform(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        return (data - self.offset) / self.coefficient

    def compute_jacobian(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        if not isinstance(self.coefficient, ndarray):
            return self.coefficient * eye(data.shape[-1])
        else:
            return diag(self.coefficient)

    def compute_jacobian_inverse(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        if not isinstance(self.coefficient, ndarray):
            return 1 / self.coefficient * eye(data.shape[-1])
        else:
            return diag(1 / self.coefficient)
