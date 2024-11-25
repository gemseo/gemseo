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
"""Settings for the augmented lagrangian algorithm."""

from __future__ import annotations

from collections.abc import Iterable  # noqa:TC003
from functools import partial
from typing import Annotated
from typing import Any
from typing import Callable

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa:TC002
from pydantic import WithJsonSchema

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.typing import StrKeyMapping  # noqa:TC001
from gemseo.utils.pydantic import copy_field

copy_field_opt = partial(copy_field, model=BaseOptimizerSettings)


class BaseAugmentedLagragianSettings(BaseOptimizerSettings):
    """The base augmented lagrangian settings."""

    initial_rho: NonNegativeFloat = Field(
        default=10.0,
        description="""The initial penalty value.""",
    )

    sub_algorithm_name: str = Field(
        description="""The name of the optimizer used to solve each sub-problem."""
    )

    sub_algorithm_settings: StrKeyMapping = Field(
        default_factory=dict,
        description="""The settings of the optimizer used to solve each sub-problem.""",
    )

    sub_problem_constraints: Iterable[str] = Field(
        default=(),
        description="""The constraints to keep in the sub-problem.

If ``empty``, all constraints are handled by the Augmented Lagrangian method
which implies that the sub-problem is unconstrained.""",
    )

    update_options_callback: (
        Annotated[Callable[[Any], Any], WithJsonSchema({})] | None
    ) = Field(
        default=None,  # Default is None since it's now exclusively a callable
        description="""A callable for updating parameters or a function call.""",
    )

    ftol_rel: NonNegativeFloat = copy_field_opt("ftol_rel", default=1e-9)

    ftol_abs: NonNegativeFloat = copy_field_opt("ftol_abs", default=1e-9)

    xtol_rel: NonNegativeFloat = copy_field_opt("xtol_rel", default=1e-9)

    xtol_abs: NonNegativeFloat = copy_field_opt("xtol_abs", default=1e-9)
