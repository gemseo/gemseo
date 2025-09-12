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
"""Base settings of functional chaos expansion model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import PositiveInt
from pydantic import model_validator

from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,
)

if TYPE_CHECKING:
    from typing_extensions import Self


class BaseFCERegressor_Settings(BaseRegressorSettings):  # noqa: N801
    """Base settings for functional chaos expansion (FCE) models."""

    degree: PositiveInt = Field(
        default=2, description="The maximum total degree of the FCE."
    )

    learn_jacobian_data: bool = Field(
        default=False,
        description="""Whether to learn the Jacobian data from training dataset.

The training dataset cannot contain both Jacobian data and special Jacobian data.

The options ``learn_jacobian_data`` and ``use_special_jacobian_data``
 are not compatible.""",
    )

    use_special_jacobian_data: bool = Field(
        default=False,
        description="""Whether to use the special Jacobian data from training dataset.

Special Jacobian data are samples of partial derivatives
with respect to variables that are not inputs of the FCE.

The training dataset cannot contain both Jacobian data and special Jacobian data.

The options ``use_special_jacobian_data`` and ``learn_jacobian_data``
 are not compatible.""",
    )

    @model_validator(mode="after")
    def __check_jacobian_options(self) -> Self:
        """Check the compatibility of Jacobian usage options."""
        if self.learn_jacobian_data and self.use_special_jacobian_data:
            msg = (
                "Only one of the options "
                "learn_jacobian_data and use_special_jacobian_data can be True."
            )
            raise ValueError(msg)

        return self
