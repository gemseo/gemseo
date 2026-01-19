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
"""Settings for the supervised machine learning models."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

from pydantic import Field

from gemseo.mlearning.core.models.ml_model_settings import BaseMLModelSettings


class BaseMLSupervisedModelSettings(BaseMLModelSettings):
    """The settings common to all the supervised machine learning models."""

    input_names: Sequence[str] = Field(
        default=(), description="The names of the input variables"
    )

    output_names: Sequence[str] = Field(
        default=(), description="The names of the output variables"
    )
