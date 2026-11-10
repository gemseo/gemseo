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
"""Settings for the SciPy basin-hopping algorithm."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated
from typing import ClassVar
from typing import Literal

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa:TC002
from pydantic import PositiveFloat  # noqa:TC002
from pydantic import PositiveInt  # noqa:TC002
from pydantic import WithJsonSchema

from gemseo.optimization.scipy_global.settings.base_scipy_global_settings import (
    BaseSciPyGlobalSettings,
)
from gemseo.optimization.scipy_global.settings.basinhopping_local_method import (
    BasinHoppingLocalMethod,
)
from gemseo.util import seeder
from gemseo.util.typing import StrKeyMapping  # noqa: TC001


class BASINHOPPING_Settings(BaseSciPyGlobalSettings):  # noqa: N801
    """The SciPy basin-hopping settings."""

    temperature: NonNegativeFloat = Field(
        default=1.0,
        description="""The temperature of the Metropolis acceptance criterion.

Use higher values to accept more often a local minimum worse than the current one.
The temperature should be comparable to the typical difference between local minima
values.""",
    )

    stepsize: PositiveFloat = Field(
        default=0.5,
        description="""The initial maximum displacement along each coordinate.

It is adapted during the run to reach `target_accept_rate`.
It is expressed in the normalized design space when `normalize_design_space` is
`True`.""",
    )

    interval: PositiveInt = Field(
        default=50,
        description="""The number of iterations between two step size updates.""",
    )

    target_accept_rate: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="""The target acceptance rate used to adapt the step size.

The step size increases when the acceptance rate is above this target
and decreases otherwise.""",
    )

    stepwise_factor: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description="""The factor by which the step size is multiplied or divided.""",
    )

    seed: int = Field(
        default=seeder.seed,
        description="""The random seed.""",
    )

    disp: bool = Field(
        default=False,
        description="""Whether to print the status messages of SciPy.""",
    )

    niter_success: PositiveInt | None = Field(
        default=None,
        description="""The number of iterations without improving the best local minimum
after which to stop.

If `None`, do not use this stopping criterion.""",
    )

    accept_test: (
        Annotated[Callable[..., bool | Literal["force accept"]], WithJsonSchema({})]
        | None
    ) = Field(
        default=None,
        description="""A test to accept or reject a local minimum.

It is called with the keyword arguments `f_new`, `x_new`, `f_old` and `x_old`,
where `f_new` and `x_new` are the objective value and the point of the new local
minimum,
and `f_old` and `x_old` those of the current one,
or with the keyword arguments `res_new` and `res_old`,
the SciPy `OptimizeResult` of these local minimizations.
It returns `False` to reject the new local minimum,
`True` to submit it to the Metropolis acceptance criterion
and `"force accept"` to accept it whatever this criterion says,
e.g. to escape from a local minimum.
The points are expressed in the normalized design space when
`normalize_design_space` is `True`.

If `None`, use the Metropolis acceptance criterion only.""",
    )

    local_method: BasinHoppingLocalMethod | None = Field(
        default=None,
        description="""The local minimization method.

If `None`,
when the objective and the constraints have gradients
or when the `differentiation_method` approximates them
(finite differences or complex step),
use `"SLSQP"` for a problem with constraints and `"L-BFGS-B"` otherwise;
when they do not,
use `"COBYLA"` for a problem with inequality constraints only,
`"COBYQA"` for a problem with equality constraints
and `"Nelder-Mead"` for a problem without constraints.
Problems with constraints require `"SLSQP"`, `"COBYLA"` (inequality constraints
only), `"COBYQA"` or `"trust-constr"`.""",
    )

    local_options: StrKeyMapping = Field(
        default_factory=dict,
        description="""The options of the local minimization method.""",
    )

    _redundant_settings: ClassVar[list[str]] = [
        "niter",
        "minimizer_kwargs",
        "rng",
        "T",
        "take_step",
    ]
