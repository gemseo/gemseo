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
"""Settings for the SciPy TNC algorithm."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TCH002

from gemseo.algos.opt.gradient_based_algorithm_settings import (
    GradientBasedAlgorithmSettings,
)
from gemseo.algos.opt.scipy_local._base_scipy_local_settings import (
    BaseScipyLocalSettings,
)


class TNCSettings(BaseScipyLocalSettings, GradientBasedAlgorithmSettings):
    """Settings for the SciPy TNC algorithm."""

    eps: NonNegativeFloat = Field(
        default=1e-8,
        description=(
            """The absolute step size forforward differences.

            The forward differences is used to approximate the Jacobian if not provided.
            """
        ),
    )

    scale: list[float] | None = Field(
        default=None,
        description=(
            """The scaling factors to apply to each variable.
            If ``None``, the factors are up-low for interval bounded variables and
            1+|x] for the others.
            """
        ),
    )

    offset: float | None = Field(
        default=None,
        description=(
            """The value to subtract from each variable.
            If ``None``, the offsets are (up+low)/2 for interval bounded variables and
            x for the others.
            """
        ),
    )

    maxCGit: int = Field(  # noqa: N815
        default=-1,
        description=(
            """The maximum number of hessian-vector evaluations per main iteration."""
        ),
    )

    eta: float = Field(
        default=-1,
        description="""The severity of the line search.""",
    )

    stepmx: NonNegativeFloat = Field(
        default=0.0,
        description=(
            """The maximum step for the line search (may be increased during call)."""
        ),
    )

    accuracy: NonNegativeFloat = Field(
        default=0.0,
        description="""The relative precision for finite difference calculations.""",
    )

    minfev: float = Field(
        default=0.0,
        description="""The minimum function value estimate.""",
    )

    gtol: NonNegativeFloat = Field(
        default=1e-6,
        description=(
            "The precision goal for the projected gradient value to stop the algorithm."
        ),
    )

    rescale: NonNegativeFloat = Field(
        default=1.3,
        description=(
            "The log10 scaling factor used to trigger the objectiv function rescaling."
        ),
    )