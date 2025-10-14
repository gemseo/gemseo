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

"""Settings for the scikit-learn elastic net algorithm with built-in cross-validation."""  # noqa: E501

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from pydantic import NonNegativeFloat

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)
from gemseo.mlearning.linear_model_fitting.elastic_net_settings import _ElasticNetMixin


class ElasticNetCV_Settings(_ElasticNetMixin, BaseLinearModelFitter_Settings):  # noqa: N801
    """Settings for the scikit-learn elastic net algorithm with built-in cross-validation."""  # noqa: E501

    _TARGET_CLASS_NAME: ClassVar[str] = "ElasticNetCV"

    alphas: tuple[NonNegativeFloat, ...] = Field(
        default=(0.001, 0.01, 0.1, 1.0, 10.0),
        description=r"""Values of :math:`\alpha` to try.
The constant :math:`\alpha` multiplies the L1 and 2 terms,
controlling regularization strength.""",
    )

    l1_ratio: tuple[NonNegativeFloat, ...] = Field(
        default=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1),
        description=r"""Values of :math:`\rho` to try.
The ElasticNet mixing parameter :math:`\rho`.
For ``l1_ratio = 0``, the penalty is an L2 penalty.
For ``l1_ratio = 1``, it is an L1 penalty.
For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.""",
    )

    cv: int | None = Field(
        default=None,
        description="""The number of folds.
If ``None``, use the efficient Leave-One-Out cross-validation.""",
    )
