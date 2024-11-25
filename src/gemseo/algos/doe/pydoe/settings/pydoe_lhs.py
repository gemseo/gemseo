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
"""Settings for the LHS DOE from the pyDOE library."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveInt  # noqa: TC002
from strenum import StrEnum

from gemseo.algos.doe.pydoe.settings.base_pydoe_settings import BasePyDOESettings


class Criterion(StrEnum):
    """The criteria for the LHS."""

    center = "center"
    c = "c"
    maximin = "maximin"
    m = "m"
    centermaximin = "centermaximin"
    cm = "cm"
    correlation = "correlation"
    corr = "corr"
    lhsmu = "lhsmu"


class PYDOE_LHS_Settings(BasePyDOESettings):  # noqa: N801
    """The settings for the LHS DOE from the pyDOE library."""

    _TARGET_CLASS_NAME = "PYDOE_LHS"

    criterion: Criterion | None = Field(
        default=None,
        description="""The criterion to use when sampling the points.

If ``None``, randomize the points within the intervals.""",
    )

    iterations: PositiveInt = Field(
        default=5,
        description="The number of iterations in the ``correlation``/``maximin`` algorithms.",  # noqa: E501
    )

    n_samples: PositiveInt = Field(description="""The number of samples.""")

    random_state: PositiveInt | None = Field(
        default=None,
        description="""The seed used for reproducibility reasons.

If ``None``, use :class:`~.BaseDOELibrary.seed`.""",
    )
