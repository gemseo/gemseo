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

from collections.abc import Callable
from collections.abc import Iterable  # noqa:TC003
from typing import Annotated
from typing import Any
from typing import ClassVar

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa:TC002
from pydantic import WithJsonSchema

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.typing import StrKeyMapping  # noqa:TC001


class BaseAugmentedLagrangianSettings(BaseOptimizerSettings):
    """The base augmented lagrangian settings."""

    _INHERITED_FIELD_DEFAULTS: ClassVar[StrKeyMapping] = {
        "ftol_rel": 1e-9,
        "ftol_abs": 1e-9,
        "xtol_rel": 1e-9,
        "xtol_abs": 1e-9,
    }

    initial_rho: NonNegativeFloat = Field(
        default=10.0,
        description="""The initial penalty value.""",
    )

    sub_algorithm_settings: BaseOptimizerSettings = Field(
        description="""The settings of the optimizer used to solve each sub-problem.""",
    )

    sub_problem_constraints: Iterable[str] = Field(
        default=(),
        description="""The constraints to keep in the sub-problem.

If `empty`, all constraints are handled by the Augmented Lagrangian method
which implies that the sub-problem is unconstrained.""",
    )

    update_options_callback: (
        Annotated[Callable[[Any], Any], WithJsonSchema({})] | None
    ) = Field(
        default=None,  # Default is None since it's now exclusively a callable
        description="""A callable for updating parameters or a function call.""",
    )
