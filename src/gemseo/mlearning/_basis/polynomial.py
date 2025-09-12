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

"""Polynomial multivariate basis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openturns import OrthogonalProductPolynomialFactory
from openturns import StandardDistributionPolynomialFactory

from gemseo.mlearning._basis.base_ot_basis import BaseOTBasis

if TYPE_CHECKING:
    from openturns import LinearEnumerateFunction


class Polynomial(BaseOTBasis):
    """The polynomial multivariate basis."""

    def _create_full_basis(
        self, input_dimension: int, enumerate_function: LinearEnumerateFunction
    ) -> OrthogonalProductPolynomialFactory:
        return OrthogonalProductPolynomialFactory(
            [
                StandardDistributionPolynomialFactory(
                    self._distribution.distribution.getMarginal(i)
                )
                for i in range(input_dimension)
            ],
            enumerate_function,
        )
