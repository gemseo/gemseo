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
"""Settings for the SciPy GCROT algorithm."""

from __future__ import annotations

from collections.abc import Iterable  # noqa: TC003
from typing import Literal

from pydantic import Field
from pydantic import PositiveInt  # noqa: TC002

from gemseo.algos.linear_solvers.scipy_linalg.settings.base_scipy_linalg_settings import (  # noqa: E501
    BaseSciPyLinalgSettingsBase,
)
from gemseo.typing import NumberArray  # noqa: TC001


class GCROT_Settings(BaseSciPyLinalgSettingsBase):  # noqa: N801
    """The settings of the SciPy GCROT algorithm."""

    _TARGET_CLASS_NAME = "GCROT"

    m: PositiveInt = Field(
        default=30,
        description="Number of inner FGMRES iterations per each outer iteration.",
    )

    k: PositiveInt = Field(
        default=30,
        description="""Number of vectors to carry between inner FGMRES iterations.

If ``None`` use the same value as ``m``.""",
    )

    CU: Iterable[tuple[NumberArray, NumberArray]] | None = Field(
        default=None,
        description="""List of tuples `(c, u)` required to form the matrices C and U.

If ``None`` start from empty matrices.""",
    )

    discard_C: bool = Field(  # noqa: N815
        default=True,
        description="""Whether to discard the C-vectors at the end.""",
    )

    truncate: Literal["oldest", "smallest"] = Field(
        default="oldest", description="The vectors from the previous cycle to drop."
    )
