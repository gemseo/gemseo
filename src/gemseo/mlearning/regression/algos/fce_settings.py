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
"""Settings of the functional chaos expansion model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import model_validator
from strenum import StrEnum

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,  # noqa: TC001
)
from gemseo.mlearning.linear_model_fitting.omp_settings import (
    OrthogonalMatchingPursuit_Settings,
)
from gemseo.mlearning.regression.algos.base_fce_settings import (
    BaseFCERegressor_Settings,
)

if TYPE_CHECKING:
    from typing_extensions import Self


class OrthonormalFunctionBasis(StrEnum):
    """An orthonormal function basis."""

    POLYNOMIAL = "Polynomial"
    FOURIER = "Fourier"
    HAAR = "Haar"


class FCERegressor_Settings(BaseFCERegressor_Settings):  # noqa: N801
    """The settings of the functional chaos expansion model."""

    _TARGET_CLASS_NAME = "FCERegressor"

    linear_model_fitter_settings: BaseLinearModelFitter_Settings | None = Field(
        default=None,
        description="""The settings of the linear solver.
If ``None``, use the default :class:`.OrthogonalMatchingPursuit_Settings`.""",
    )

    basis: OrthonormalFunctionBasis = Field(
        default=OrthonormalFunctionBasis.POLYNOMIAL,
        description="The orthonormal function basis.",
    )

    @model_validator(mode="after")
    def __check_linear_model_fitter_settings(self) -> Self:
        """Check the option linear_model_fitter_settings."""
        if self.linear_model_fitter_settings is None:
            self.linear_model_fitter_settings = OrthogonalMatchingPursuit_Settings()

        return self
