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
"""The centered LHS algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.algos.doe._openturns.ot_standard_lhs import OTStandardLHS

if TYPE_CHECKING:
    from numpy import ndarray


class OTCenteredLHS(OTStandardLHS):
    """The centered LHS algorithm.

    .. note:: This class is a singleton.
    """

    def generate_samples(  # noqa: D102
        self, n_samples: int, dimension: int, **options: Any
    ) -> ndarray:
        samples = super().generate_samples(n_samples, dimension)
        return (samples // (1.0 / n_samples) + 0.5) / n_samples
