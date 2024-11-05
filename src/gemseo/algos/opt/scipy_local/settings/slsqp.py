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

from typing import ClassVar

from pydantic import Field

from gemseo.algos.opt.base_gradient_based_algorithm_settings import (
    BaseGradientBasedAlgorithmSettings,
)
from gemseo.algos.opt.scipy_local.settings.base_scipy_local_settings import (
    BaseScipyLocalSettings,
)


class SLSQP_Settings(BaseScipyLocalSettings, BaseGradientBasedAlgorithmSettings):  # noqa: N801
    """Settings for the SciPy SLSQP algorithm."""

    _TARGET_CLASS_NAME = "SLSQP"

    iprint: int = Field(
        default=-1,
        description="""The flag to control the frequency of output.

Default is no output.""",
    )

    _redundant_settings: ClassVar[list[str]] = ["maxiter", "eps"]
