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

"""Settings for the scikit-lear least angle regression (LARS) algorithm with build-in cross-validation."""  # noqa: E501

from __future__ import annotations

from typing import ClassVar
from typing import Literal

from numpy import finfo
from numpy import ndarray  # noqa: TC002
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)


class LARSCV_Settings(BaseLinearModelFitter_Settings):  # noqa: N801
    """Settings for the scikit-learn least angle regression (LARS) algorithm with build-in cross-validation."""  # noqa: E501

    _TARGET_CLASS_NAME: ClassVar[str] = "LARSCV"

    copy_X: bool = Field(  # noqa: N815
        default=True,
        description="""If ``True``, input data will be copied;
else, it may be overwritten""",
    )

    max_iter: PositiveInt = Field(
        default=500, description="The maximum number of iterations."
    )

    cv: int | None = Field(
        default=None,
        description="""The number of folds.
If ``None``, use the efficient Leave-One-Out cross-validation.""",
    )

    eps: NonNegativeFloat = Field(
        default=finfo(float).eps,
        description="""The machine-precision regularization
in the computation of the Cholesky diagonal factors.
Increase this for very ill-conditioned systems.
Unlike the ``tol`` parameter in some iterative optimization-based algorithms,
this parameter does not control the tolerance of the optimization.""",
    )

    precompute: Literal["auto"] | ndarray = Field(
        default="auto",
        description="""Whether to use a precomputed Gram matrix
to speed up calculations. If set to "auto" let us decide.
The Gram matrix can also be passed as argument.""",
    )

    verbose: bool = Field(default=False, description="Sets the verbosity amount.")
