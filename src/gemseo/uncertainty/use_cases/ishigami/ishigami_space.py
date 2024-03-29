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
"""The uncertain space used in the Ishigami use case."""

from __future__ import annotations

from numpy import pi
from strenum import StrEnum

from gemseo.algos.parameter_space import ParameterSpace


class IshigamiSpace(ParameterSpace):
    r"""The uncertain space used in the Ishigami use case.

    :math:`X_1,X_2,X_3` are independent random variables
    uniformly distributed between :math:`-\pi` and :math:`\pi`.

    This uncertain space uses the class :class:`.SPUniformDistribution`.

    See :cite:`ishigami1990`.
    """

    class UniformDistribution(StrEnum):
        """The name of the class implementing the uniform distribution."""

        OPENTURNS = "OTUniformDistribution"
        SCIPY = "SPUniformDistribution"

    def __init__(
        self, uniform_distribution_name: UniformDistribution = UniformDistribution.SCIPY
    ) -> None:
        """
        Args:
            uniform_distribution_name: The name of the class
                implementing the uniform distribution.
        """  # noqa: D205, D212
        super().__init__()
        for index in range(3):
            self.add_random_variable(
                f"x{index + 1}", uniform_distribution_name, minimum=-pi, maximum=pi
            )
