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
"""Settings of the Gaussian process regressor from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003
from typing import Annotated
from typing import Callable

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import WithJsonSchema
from sklearn.gaussian_process.kernels import Kernel  # noqa: TC002

from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,
)
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC001
from gemseo.utils.seeder import SEED


class GaussianProcessRegressor_Settings(BaseRegressorSettings):  # noqa: N801
    """The settings of the Gaussian process regressor from scikit-learn."""

    kernel: Annotated[Kernel, WithJsonSchema({})] | None = Field(
        default=None,
        description="""The kernel specifying the covariance model.

If ``None``, use a Matérn(2.5).""",
    )

    bounds: tuple | tuple[float, float] | Mapping[str, tuple[float, float]] = Field(
        default=(),
        description="""The lower and upper bounds of the length scales.

Either a unique lower-upper pair common to all the inputs
or lower-upper pairs for some of them.
When ``bounds`` is empty or when an input has no pair,
the lower bound is 0.01 and the upper bound is 100.

This argument is ignored when ``kernel`` is ``None``.""",
    )

    alpha: float | NDArrayPydantic = Field(
        default=1e-10, description="The nugget effect to regularize the model."
    )

    optimizer: str | Annotated[Callable, WithJsonSchema({})] = Field(
        default="fmin_l_bfgs_b",
        description="The optimization algorithm to find the parameter length scales.",
    )

    n_restarts_optimizer: NonNegativeInt = Field(
        default=10, description="The number of restarts of the optimizer."
    )

    random_state: NonNegativeInt | None = Field(
        default=SEED,
        description="""The random state parameter.

If ``None``, use the global random state instance from ``numpy.random``.
Creating the model multiple times will produce different results.
If ``int``, use a new random number generator seeded by this integer.
This will produce the same results.""",
    )
