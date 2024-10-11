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
"""Settings for the SciPy SLSQP algorithm."""

from __future__ import annotations

from pydantic import Field

from gemseo.algos.opt._gradient_based_algorithm_settings import (
    GradientBasedAlgorithmSettings,
)
from gemseo.algos.opt.scipy_local._base_scipy_local_settings import (
    BaseScipyLocalSettings,
)


class SLSQPSettings(BaseScipyLocalSettings, GradientBasedAlgorithmSettings):
    """Settings for the SciPy SLSQP algorithm."""

    iprint: int = Field(
        default=-1,
        description=(
            """The flag to control the frequency of output.

            Default is no output.
            """
        ),
    )
