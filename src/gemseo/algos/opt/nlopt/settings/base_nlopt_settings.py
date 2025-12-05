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
from math import inf
from typing import ClassVar

from pydantic import Field

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.typing import StrKeyMapping
from gemseo.utils.pydantic import copy_field

copy_field_opt = partial(copy_field, model=BaseOptimizerSettings)  # pyright: ignore[reportUndefinedVariable]


class BaseNLoptSettings(BaseOptimizerSettings):
    """The NLopt optimization library setting."""

    _INHERITED_FIELD_DEFAULTS: ClassVar[StrKeyMapping] = {
        "ftol_rel": 1e-8,
        "ftol_abs": 1e-14,
        "stop_crit_n_x": None,
        "xtol_rel": 1e-8,
        "xtol_abs": 1e-14,
    }
    _INHERITED_FIELD_TYPES: ClassVar[StrKeyMapping] = {"stop_crit_n_x": int | None}

    stopval: float = Field(
        default=-inf,
        description="""The objective value at which the optimization will stop.""",
    )
