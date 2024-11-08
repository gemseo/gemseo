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
"""Settings for the SciPy GMRES algorithm."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveInt  # noqa: TCH002

from gemseo.algos.linear_solvers.scipy_linalg.settings.base_scipy_linalg_settings import (  # noqa: E501
    BaseSciPyLinalgSettingsBase,
)


class GMRES_Settings(BaseSciPyLinalgSettingsBase):  # noqa: N801
    """The settings of the SciPy GMRES algorithm."""

    _TARGET_CLASS_NAME = "GMRES"

    restart: PositiveInt = Field(
        default=20,
        description="""Number of iterations between restarts.""",
    )
