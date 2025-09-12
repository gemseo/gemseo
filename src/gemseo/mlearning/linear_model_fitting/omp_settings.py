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

"""Settings for the scikit-learn Orthogonal Matching Pursuit (OMP) algorithm."""

from __future__ import annotations

from typing import ClassVar
from typing import Literal

from pydantic import Field
from pydantic import PositiveFloat
from pydantic import PositiveInt

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)


class OrthogonalMatchingPursuit_Settings(BaseLinearModelFitter_Settings):  # noqa: N801
    """Settings for the scikit-learn Orthogonal Matching Pursuit (OMP) algorithm."""

    _TARGET_CLASS_NAME: ClassVar[str] = "OrthogonalMatchingPursuit"

    n_nonzero_coefs: PositiveInt | None = Field(
        default=None,
        description="""The desired number of non-zero coefficients.
Ignored if :attr:`.tol` is set.
When ``None`` and :attr:`.tol` is also ``None``,
this value is either set to 10% of the input dimension or 1, whichever is greater.""",
    )

    precompute: bool | Literal["auto"] = Field(
        default="auto",
        description="""Whether to use a precomputed Gram and Xy matrix
to speed up calculations.
Improves performance
when the output dimension or the number of samples is very large.""",
    )

    tol: PositiveFloat = Field(
        default=1e-7,
        description="""The maximum squared norm of the residual
normalized by the infinite norm of the output data.""",
    )
