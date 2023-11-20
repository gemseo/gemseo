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
"""The Monte Carlo algorithm."""

from __future__ import annotations

from typing import Any

from numpy import array
from numpy import ndarray

from gemseo.algos.doe._openturns.base_ot_doe import BaseOTDOE


class OTMonteCarlo(BaseOTDOE):
    """The Monte Carlo algorithm.

    .. note:: This class is a singleton.
    """

    def generate_samples(  # noqa: D102
        self, n_samples: int, dimension: int, **options: Any
    ) -> ndarray:
        samples = self._STANDARD_UNIFORM_DISTRIBUTION.getSample(dimension * n_samples)
        return array(samples).reshape((n_samples, dimension))
