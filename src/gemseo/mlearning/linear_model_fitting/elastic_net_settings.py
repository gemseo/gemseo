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

"""Settings for the scikit-learn elastic net algorithm."""

from __future__ import annotations

from typing import ClassVar
from typing import Literal

from numpy import ndarray  # noqa: TC002
from numpy.random import RandomState  # noqa: TC002
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveFloat
from pydantic import PositiveInt

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)
from gemseo.settings.base_settings import BaseSettings


class _ElasticNetMixin(BaseSettings):
    """Mixin for defining the settings of the scikit-learn elasticnet algorithm."""

    copy_X: bool = Field(  # noqa: N815
        default=True,
        description="""If ``True``, input data will be copied;
else, it may be overwritten""",
    )

    max_iter: PositiveInt = Field(
        default=1000, description="""The maximum number of iterations."""
    )

    positive: bool = Field(
        default=False,
        description="""When set to ``True``, forces the coefficients to be positive.""",
    )

    precompute: bool | ndarray = Field(
        default=False,
        description="""Whether to use a precomputed Gram matrix
to speed up calculations.
The Gram matrix can also be passed as ``precompute`` value.
For sparse input this option is always ``False`` to preserve sparsity.""",
    )

    random_state: int | RandomState | None = Field(
        default=None,
        description="""The seed of the pseudo random number generator
that selects a random feature to update.
Used when ``selection == "random"``.
Pass an int for reproducible output across multiple function calls.""",
    )

    selection: Literal["cyclic", "random"] = Field(
        default="cyclic",
        description="""If set to "random",
a random coefficient is updated every iteration
rather than looping over features sequentially by default.
This (setting to "random") often leads to significantly faster convergence
especially when ``tol`` is higher than 1e-4.""",
    )

    tol: PositiveFloat = Field(
        default=1e-4,
        description="""The tolerance for the optimization:
if the updates are smaller than ``tol``,
the optimization code checks the dual gap for optimality
and continues until it is smaller than ``tol``.""",
    )


class ElasticNet_Settings(_ElasticNetMixin, BaseLinearModelFitter_Settings):  # noqa: N801
    """Settings for the scikit-learn elastic net algorithm."""

    _TARGET_CLASS_NAME: ClassVar[str] = "ElasticNet"

    alpha: NonNegativeFloat = Field(
        default=1.0,
        description=r"""The constant :math:`\alpha` that multiplies the L1 and 2 terms,
controlling regularization strength.""",
    )

    l1_ratio: NonNegativeFloat = Field(
        default=0.5,
        le=1.0,
        description=r"""The ElasticNet mixing parameter :math:`\rho`.
    For ``l1_ratio = 0``, the penalty is an L2 penalty.
    For ``l1_ratio = 1``, it is an L1 penalty.
    For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.""",
    )

    warm_start: bool = Field(
        default=False,
        description="""When set to ``True``,
reuse the solution of the previous call to fit as initialization,
otherwise, just erase the previous solution.""",
    )
