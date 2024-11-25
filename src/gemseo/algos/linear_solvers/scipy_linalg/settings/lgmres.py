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
"""Settings for the SciPy LGMRES algorithm."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveInt  # noqa: TC002

from gemseo.algos.linear_solvers.scipy_linalg.settings.base_scipy_linalg_settings import (  # noqa: E501
    BaseSciPyLinalgSettingsBase,
)
from gemseo.typing import NumberArray  # noqa: TC001


class LGMRES_Settings(BaseSciPyLinalgSettingsBase):  # noqa: N801
    """The settings of the SciPy LGMRES algorithm."""

    _TARGET_CLASS_NAME = "LGMRES"

    inner_m: PositiveInt = Field(
        default=30,
        description="""Number of inner GMRES iterations per each outer iteration.""",
    )

    outer_k: PositiveInt = Field(
        default=3,
        description="""Number of vectors to carry between inner GMRES iterations.""",
    )

    outer_v: list[tuple[NumberArray, NumberArray]] = Field(
        default_factory=list,
        description="""List of tuples `(v, Av)` used to augment the Krylov subspace.""",
    )

    store_outer_Av: bool = Field(  # noqa: N815
        default=True,
        description=(
            """Whether A @ v should be additionnaly stored in `outer_v` list."""
        ),
    )

    prepend_outer_v: bool = Field(
        default=False,
        description=(
            """Whether to put `outer_v` augmentation vectors before Krylov iterates."""
        ),
    )


DEFAULTSettings = LGMRES_Settings
