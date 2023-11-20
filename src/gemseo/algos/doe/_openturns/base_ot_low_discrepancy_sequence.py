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
"""The low-discrepancy sequence algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import array
from numpy import ndarray

from gemseo.algos.doe._openturns.base_ot_doe import BaseOTDOE

if TYPE_CHECKING:
    from openturns import LowDiscrepancySequenceImplementation


class BaseOTLowDiscrepancySequence(BaseOTDOE):
    """The low-discrepancy sequence algorithm."""

    _ALGO_CLASS: ClassVar[type[LowDiscrepancySequenceImplementation]]

    def generate_samples(  # noqa: D102
        self, n_samples: int, dimension: int, **options: Any
    ) -> ndarray:
        return array(self._ALGO_CLASS(dimension).generate(n_samples))
