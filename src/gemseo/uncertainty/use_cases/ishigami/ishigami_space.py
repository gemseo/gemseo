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

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.utils.python_compatibility import Final

_DISTRIBUTION_NAME: Final[str] = "SPUniformDistribution"
"""The probability distribution for the different random variables."""


class IshigamiSpace(ParameterSpace):
    r"""The uncertain space used in the Ishigami use case.

    :math:`X_1,X_2,X_3` are independent random variables
    uniformly distributed between :math:`-\pi` and :math:`\pi`.

    This uncertain space uses the class :class:`.SPUniformDistribution`.

    See :cite:`ishigami1990`.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        for index in range(3):
            self.add_random_variable(
                f"x{index+1}", _DISTRIBUTION_NAME, minimum=-pi, maximum=pi
            )
