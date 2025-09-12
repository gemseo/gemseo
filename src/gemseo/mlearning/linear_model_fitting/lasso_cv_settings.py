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

"""Settings for the scikit-learn lasso algorithm with built-in cross-validation."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from pydantic import NonNegativeFloat

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)
from gemseo.mlearning.linear_model_fitting.lasso_settings import _LassoSettingsMixin


class LassoCV_Settings(_LassoSettingsMixin, BaseLinearModelFitter_Settings):  # noqa: N801
    """Settings for the scikit-learn lasso algorithm with built-in cross-validation."""

    _TARGET_CLASS_NAME: ClassVar[str] = "LassoCV"

    alphas: tuple[NonNegativeFloat, ...] = Field(
        default=(0.001, 0.01, 0.1, 1.0, 10.0),
        description=r"""Values of :math:`\alpha` to try.
The constant :math:`\alpha` multiplies the L1 term,
controlling regularization strength.""",
    )

    cv: int | None = Field(
        default=None,
        description="""The number of folds.
If ``None``, use the efficient Leave-One-Out cross-validation.""",
    )
