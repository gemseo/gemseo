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
"""Settings for the SciPy algorithms."""

from __future__ import annotations

from functools import partial

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa:TC002

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.utils.pydantic import copy_field

copy_field_opt = partial(copy_field, model=BaseOptimizerSettings)


class BaseScipyLocalSettings(BaseOptimizerSettings):
    """The SciPy local optimization library setting."""

    disp: bool = Field(
        default=False,
        description="""Whether to print convergence messages.""",
    )

    eq_tolerance: NonNegativeFloat = copy_field_opt("eq_tolerance", default=1e-6)

    ftol_rel: NonNegativeFloat = copy_field_opt("ftol_rel", default=1e-9)

    ftol_abs: NonNegativeFloat = copy_field_opt("ftol_abs", default=1e-9)

    xtol_rel: NonNegativeFloat = copy_field_opt("xtol_rel", default=1e-9)

    xtol_abs: NonNegativeFloat = copy_field_opt("xtol_abs", default=1e-9)
