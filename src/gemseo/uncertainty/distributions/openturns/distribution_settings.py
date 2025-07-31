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
"""Settings for the OpenTURNS-based probability distributions."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003
from typing import TYPE_CHECKING
from typing import Final

from pydantic import Field
from pydantic import model_validator

from gemseo import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from typing_extensions import Self

from gemseo.uncertainty.distributions.base_distribution_settings import (
    BaseDistribution_Settings,
)

_INTERFACED_DISTRIBUTION: Final[str] = "Uniform"
"""The default value of interfaced_distribution."""

_LOWER_BOUND: Final[None] = None
"""The default value of lower_bound."""

_PARAMETERS: Final[()] = ()
"""The default value of parameters."""

_STANDARD_PARAMETERS: Final[READ_ONLY_EMPTY_DICT] = READ_ONLY_EMPTY_DICT
"""The default value of standard_parameters."""

_TRANSFORMATION: Final[str] = ""
"""The default value of transformation."""

_THRESHOLD: Final[float] = 0.5
"""The default value of threshold."""

_UPPER_BOUND: Final[None] = None
"""The default value of upper_bound."""


class _OTDistribution_Settings_Mixin:  # noqa: N801
    """A mixin for the settings of an OpenTURNS-based distribution."""

    transformation: str = Field(
        default=_TRANSFORMATION,
        description=r"""A transformation applied
to the random variable, e.g. :math:`\sin(x)`. If empty, no transformation.""",
    )

    lower_bound: float | None = Field(
        default=_LOWER_BOUND,
        description="""A lower bound to truncate the probability distribution.

If ``None``, no lower truncation.""",
    )

    upper_bound: float | None = Field(
        default=_UPPER_BOUND,
        description="""An upper bound to truncate the probability distribution.

If ``None``, no upper truncation.""",
    )

    threshold: float = Field(
        default=_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=r"""A truncation threshold in :math:`[0,1]`
(`see OpenTURNS documentation
<http://openturns.github.io/openturns/latest/user_manual/
_generated/openturns.TruncatedDistribution.html>`_).""",
    )

    @model_validator(mode="after")
    def __validate(self) -> Self:
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.upper_bound <= self.lower_bound
        ):
            msg = (
                "The upper truncation bound of a probability distribution must be "
                "greater than its lower truncation bound."
            )
            raise ValueError(msg)

        return self


class OTDistribution_Settings(  # noqa: N801
    BaseDistribution_Settings, _OTDistribution_Settings_Mixin
):
    """The settings of an OpenTURNS-based distribution."""

    _TARGET_CLASS_NAME = "OTDistribution"

    interfaced_distribution: str = Field(
        default="Uniform", description="The name of the probability distribution."
    )

    parameters: tuple[float, ...] = Field(
        default_factory=tuple,
        description="The parameters of the probability distribution.",
    )

    standard_parameters: Mapping[str, str | int | float] = Field(
        default_factory=dict,
        description="""The parameters of the probability distribution
used for string representation only.""",
    )
