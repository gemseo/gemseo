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

"""Settings for the scikit-learn ridge algorithm with built-in cross-validation."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import ClassVar

from pydantic import Field
from pydantic import NonNegativeFloat
from strenum import StrEnum

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)


class GCVMode(StrEnum):
    """Flag indicating the strategy for Leave-One-Out Cross-Validation."""

    AUTO = "auto"
    SVD = "svd"
    EIGEN = "eigen"


class RidgeCV_Settings(BaseLinearModelFitter_Settings):  # noqa: N801
    """Settings for the scikit-learn ridge algorithm with build-in cross validation."""

    _TARGET_CLASS_NAME: ClassVar[str] = "RidgeCV"

    alphas: tuple[NonNegativeFloat, ...] = Field(
        default=(0.001, 0.01, 0.1, 1.0, 10.0),
        description=r"""Values of :math:`\alpha` to try.
The constant :math:`\alpha` multiplies the L2 term,
controlling regularization strength.""",
    )

    gcv_mode: GCVMode = Field(
        default=GCVMode.AUTO,
        description="""The flag indicating which strategy to use
when performing Leave-One-Out Cross-Validation.""",
    )

    alpha_per_target: bool = Field(
        default=False,
        description="""The flag indicating whether to optimize the :math:`alpha` value
(picked from the ``alphas`` parameter list) for each target separately
(for multi-output settings: multiple prediction targets).""",
    )

    scoring: str | Callable | None = Field(
        default=None,
        description="""The scoring method to use for cross-validation.
If ``None``,
use the mean squared error when ``cv`` is ``None``
and the coefficient of determination :math:`R^2` otherwise
""",
    )

    cv: int | None = Field(
        default=None,
        description="""The number of folds.
If ``None``, use the efficient Leave-One-Out cross-validation.""",
    )
