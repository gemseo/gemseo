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

"""Settings for the scikit-learn Orthogonal Matching Pursuit (OMP) algorithm with build-in cross-validation."""  # noqa: E501

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveInt

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitterSettings,
)


class OrthogonalMatchingPursuitCV_Settings(BaseLinearModelFitterSettings):  # noqa: N801
    """Settings for the scikit-learn Orthogonal Matching Pursuit (OMP) algorithm with build-in cross-validation."""  # noqa: E501

    max_iter: PositiveInt | None = Field(
        default=None,
        description="""The maximum numbers of iterations to perform,
therefore maximum features to include.
10% of `n_features` but at least 5 if available.""",
    )

    cv: int | None = Field(
        default=None,
        description="""The number of folds.
If `None`, use the efficient Leave-One-Out cross-validation.""",
    )

    verbose: bool = Field(default=False, description="Sets the verbosity amount.")
