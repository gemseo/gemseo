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
"""Settings of the SVM regressor."""

from __future__ import annotations

from typing import Annotated
from typing import Callable

from pydantic import Field
from pydantic import WithJsonSchema

from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,
)


class SVMRegressor_Settings(BaseRegressorSettings):  # noqa: N801
    """The settings of the SVM regressor."""

    kernel: str | Annotated[Callable, WithJsonSchema({})] = Field(
        default="rbf",
        description="""The name of the kernel or a callable for the SVM.

Examples of names: "linear", "poly", "rbf", "sigmoid", "precomputed".""",
    )
