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
"""Settings for the SciPy DIRECT algorithm."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from pydantic import PositiveFloat  # noqa:TC002
from pydantic import PositiveInt  # noqa:TC002

from gemseo.optimization.scipy_global.settings.base_scipy_global_settings import (
    BaseSciPyGlobalSettings,
)
from gemseo.util.typing import StrKeyMapping  # noqa: TC001


class DIRECT_Settings(BaseSciPyGlobalSettings):  # noqa: N801
    """The SciPy DIRECT settings."""

    _inherited_field_defaults: ClassVar[StrKeyMapping] = {
        name: default
        for name, default in BaseSciPyGlobalSettings._inherited_field_defaults.items()
        if name != "max_iter"
    }

    max_iter: PositiveInt = Field(
        default=1_000,
        description="""The maximum number of iterations.

SciPy's DIRECT allocates memory in proportion to this budget
when the run starts,
about 50 bytes per iteration,
so a value larger than 10,000,000 is reduced to 10,000,000,
with a warning.""",
    )

    eps: PositiveFloat = Field(
        default=1e-4,
        description="""The minimal required improvement to split a hyperrectangle.

Small values make the search more local, large values more global.""",
    )

    locally_biased: bool = Field(
        default=True,
        description="""Whether to use the locally biased variant DIRECT-L.

Use `False` for problems with many local minima.""",
    )

    f_min: float = Field(
        default=float("-inf"),
        description="""The known global minimum of the objective, if any.""",
    )

    f_min_rtol: float = Field(
        default=1e-4,
        ge=0.0,
        le=1.0,
        description="""The relative tolerance with respect to `f_min` for stopping.

Only used when `f_min` is finite.""",
    )

    vol_tol: float = Field(
        default=1e-16,
        ge=0.0,
        le=1.0,
        description="""The volume of the hyperrectangle containing the lowest
objective value under which to stop.

It is a fraction of the volume of the design space.""",
    )

    len_tol: float = Field(
        default=1e-6,
        ge=0.0,
        le=1.0,
        description="""The half length of the hyperrectangle containing the lowest
objective value under which to stop.

This length is its largest side when `locally_biased` is `True`
and its diagonal otherwise,
normalized by the design space.""",
    )

    _redundant_settings: ClassVar[list[str]] = ["maxfun", "maxiter"]
