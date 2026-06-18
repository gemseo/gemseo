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
"""Settings for OpenTURNS optimization algorithms."""

from __future__ import annotations

from typing import Literal

from openturns import NLopt
from pydantic import BaseModel
from pydantic import Field
from pydantic import PositiveFloat
from pydantic import PositiveInt
from strenum import StrEnum


class BaseOTOptimizer(BaseModel):  # noqa: N801
    """The base class for the OpenTURNS optimizer settings.

    The subclass-specific Pydantic fields (those other than the `maximum_*` fields)
    are passed positionally to the OpenTURNS optimizer constructor.
    Their declaration order must match the OpenTURNS constructor signature.
    """

    maximum_absolute_error: PositiveFloat = Field(
        default=1e-5, description="The maximum absolute error."
    )

    maximum_calls_number: PositiveInt = Field(
        default=1000, description="The maximum number of calls."
    )

    maximum_constraint_error: PositiveFloat = Field(
        default=1e-5, description="The maximum constraint error."
    )

    maximum_iteration_number: PositiveInt = Field(
        default=100, description="The maximum number of iterations."
    )

    maximum_relative_error: PositiveFloat = Field(
        default=1e-5, description="The maximum relative error."
    )

    maximum_residual_error: PositiveFloat = Field(
        default=1e-5, description="The maximum residual error."
    )

    maximum_time_duration: PositiveFloat | Literal[-1] = Field(
        default=-1, description="The maximum time duration."
    )


class OTAbdoRackwitz(BaseOTOptimizer):  # noqa: N801
    """The settings of the Abdo-Rackwitz optimizer in OpenTURNS."""

    tau: PositiveFloat = Field(
        default=0.5, description="The multiplicative decrease of linear step."
    )

    omega: PositiveFloat = Field(default=0.0001, description="The Armijo factor.")

    smooth: PositiveFloat = Field(
        default=1.2, description="The growing factor in penalization term."
    )


class OTCobyla(BaseOTOptimizer):  # noqa: N801
    """The settings of the Cobyla optimizer in OpenTURNS."""

    rhoBeg: PositiveFloat = Field(  # noqa: N815
        default=0.1, description="The multiplicative decrease of linear step."
    )


NLoptAlgorithmName = StrEnum("NLoptAlgorithm", names=tuple(NLopt.GetAlgorithmNames()))


class OTNLopt(BaseOTOptimizer):  # noqa: N801
    """The settings of an NLopt optimizer in OpenTURNS."""

    algo_name: NLoptAlgorithmName = Field(
        default=NLoptAlgorithmName.LN_COBYLA,
        description="The name of an NLopt optimizer in OpenTURNS.",
    )
