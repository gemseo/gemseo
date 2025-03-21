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
"""Settings for the stratified DOEs from the OpenTURNS library."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

from pydantic import Field
from pydantic import NonNegativeInt

from gemseo.algos.doe.openturns.settings.base_openturns_settings import (
    BaseOpenTURNSSettings,
)


class BaseOTStratifiedDOESettings(BaseOpenTURNSSettings):
    """The settings for the stratified DOEs from the OpenTURNS library."""

    n_samples: NonNegativeInt = Field(
        default=0,
        description="""The number of samples.

If 0, set from the options.""",
    )

    levels: float | Sequence[float] = Field(
        default=(),
        description="""The levels.

In the case of axial, composite and factorial DOEs, the positions of the levels
relative to the center; the levels will be equispaced and symmetrical relative
to the center; e.g. ``[0.2, 0.8]`` in dimension 1 will generate the samples
``[0.15, 0.6, 0.75, 0.8, 0.95, 1]`` for an axial DOE; the values must be in
:math:`]0,1]`.

In the case of a full-factorial DOE, the number of levels per input direction;
if scalar, this value is applied to each input direction.
""",
    )

    centers: Sequence[float] | float = Field(
        default=0.5,
        description="""The center of DOE in the unit hypercube.

This option is available for the axial, composite and factorial DOE
algorithm. If scalar, this value is applied to each direction of the
hypercube; the values must be in :math:`]0,1[`.
""",
    )
