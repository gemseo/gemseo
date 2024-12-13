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
"""Settings for quasi-Newton MDA."""

from __future__ import annotations

from pydantic import Field
from strenum import StrEnum

from gemseo.mda.base_parallel_mda_settings import BaseParallelMDASettings


class QuasiNewtonMethod(StrEnum):
    """A quasi-Newton method."""

    ANDERSON = "anderson"
    BROYDEN1 = "broyden1"
    BROYDEN2 = "broyden2"
    DF_SANE = "df-sane"
    DIAG_BROYDEN = "diagbroyden"
    EXCITING_MIXING = "excitingmixing"
    HYBRID = "hybr"
    KRYLOV = "krylov"
    LEVENBERG_MARQUARDT = "lm"
    LINEAR_MIXING = "linearmixing"


class MDAQuasiNewton_Settings(BaseParallelMDASettings):  # noqa: N801
    """The settings for :class:`.MDAQuasiNewton`."""

    method: QuasiNewtonMethod = Field(
        default=QuasiNewtonMethod.HYBRID,
        description="""The name of the quasi-Newton method.""",
    )

    use_gradient: bool = Field(
        default=False,
        description="""Whether to use the analytic gradient of the discipline.""",
    )
