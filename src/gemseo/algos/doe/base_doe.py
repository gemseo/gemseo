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
"""The base DOE algorithm."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.utils.base_multiton import BaseABCMultiton

if TYPE_CHECKING:
    from gemseo.algos.doe.base_n_samples_based_doe_settings import (
        BaseNSamplesBasedDOESettings,
    )
    from gemseo.typing import RealArray


class BaseDOE(metaclass=BaseABCMultiton):
    """The base class for DOE algorithms."""

    # TODO: API: remove n_samples from the signature.
    @abstractmethod
    def generate_samples(
        self,
        n_samples: int,
        dimension: int,
        settings: BaseNSamplesBasedDOESettings | None = None,
    ) -> RealArray:
        """Generate samples.

        Args:
            n_samples: The number of samples.
                Ignored if ``settings`` is not `None`.
            dimension: The dimension of the sampling space.
            settings: The settings of the DOE algorithm.

        Returns:
            The samples.
        """
