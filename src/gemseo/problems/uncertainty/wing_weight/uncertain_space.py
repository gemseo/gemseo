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
"""The uncertain space of the wing weight problem."""

from __future__ import annotations

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.problems.uncertainty.utils import UniformDistribution


class WingWeightUncertainSpace(ParameterSpace):
    """The uncertain space of the wing weight problem."""

    UniformDistribution = UniformDistribution
    """The name of the class implementing the uniform distribution."""

    def __init__(
        self,
        uniform_distribution_name: UniformDistribution = UniformDistribution.SCIPY,
    ) -> None:
        """
        Args:
            uniform_distribution_name: The name of the class
                implementing the uniform distribution.
        """  # noqa: D205, D212
        super().__init__()

        data = {
            "A": (6.0, 10.0),
            "Lamda": (-10.0, 10.0),
            "Nz": (2.5, 6.0),
            "Sw": (150.0, 200.0),
            "Wdg": (1700.0, 2500.0),
            "Wfw": (220.0, 300.0),
            "Wp": (0.025, 0.08),
            "ell": (0.5, 1.0),
            "q": (16.0, 45.0),
            "tc": (0.08, 0.18),
        }

        for name, (minimum, maximum) in data.items():
            self.add_random_variable(
                name,
                uniform_distribution_name,
                minimum=minimum,
                maximum=maximum,
            )
