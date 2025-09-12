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

"""Base settings for linear model fitting algorithms."""

from __future__ import annotations

from pydantic import Field

from gemseo.settings.base_settings import BaseSettings


class BaseLinearModelFitter_Settings(BaseSettings):  # noqa: N801
    """Base settings for linear model fitting algorithms."""

    fit_intercept: bool = Field(
        default=True,
        description="""Whether to calculate the intercept :math:`b`
in the linear model :math:`y=Ax+b`.
Otherwise, it is assumed to be zero.

This option is ignored in presence of extra data, e.g., Jacobian observations;
the intercept :math:`b` is assumed to be zero.""",
    )
