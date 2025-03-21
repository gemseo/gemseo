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
"""Settings for the NLopt algorithms."""

from __future__ import annotations

from functools import partial

from numpy import inf
from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TC002

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.utils.pydantic import copy_field

copy_field_opt = partial(copy_field, model=BaseOptimizerSettings)


class BaseNLoptSettings(BaseOptimizerSettings):
    """The NLopt optimization library setting."""

    stopval: float = Field(
        default=-inf,
        description="""The objective value at which the optimization will stop.""",
    )

    ftol_rel: NonNegativeFloat = copy_field_opt("ftol_rel", default=1e-8)

    ftol_abs: NonNegativeFloat = copy_field_opt("ftol_abs", default=1e-14)

    stop_crit_n_x: int | None = copy_field_opt("stop_crit_n_x", default=None)

    xtol_rel: NonNegativeFloat = copy_field_opt("xtol_rel", default=1e-8)

    xtol_abs: NonNegativeFloat = copy_field_opt("xtol_abs", default=1e-14)
