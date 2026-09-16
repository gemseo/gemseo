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
"""Random variable."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING
from typing import Final

from pydantic import Field
from pydantic import field_validator

from gemseo.space.variable.base import DataType
from gemseo.space.variable.numeric import BaseNumericVariable
from gemseo.uncertainty.distribution.core.base_settings import BaseDistributionSettings
from gemseo.uncertainty.distribution.factory import distribution_factory
from gemseo.util._numpy import freeze_array
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from gemseo.uncertainty.distribution.core.base_joint import BaseJointDistribution
    from gemseo.util.typing import RealArray

_distribution_settings_tag: Final[str] = "distribution_settings"
"""The tag for the settings of the marginal probability distributions."""


def _check_distribution_libraries(library_names: set[str]) -> None:
    """Check that probability distributions come from a single library.

    Args:
        library_names: The names of the libraries
            implementing the probability distributions.

    Raises:
        ValueError: When the probability distributions
            come from more than one library.
    """
    if len(library_names) > 1:
        msg = (
            "A random space cannot mix probability distributions "
            "based on different libraries; "
            f"got {pretty_repr(library_names)}."
        )
        raise ValueError(msg)


class RandomVariable(BaseNumericVariable):
    """A random variable.

    A random variable is defined by the settings
    of the marginal probability distributions of its components,
    which are its only input;
    everything else is derived from them and read-only:
    the joint probability distribution of its components,
    its size (the number of these components),
    its data type (always float)
    and its bounds (the limits of the support of its distribution).

    Note:
        A random variable is its own copy and its own deep copy,
        so the joint probability distribution it wraps is shared
        with every copy of the variable.
        This is safe because a random variable is frozen
        and because this distribution is only read
        (statistics, support, marginals);
        the samples of a space of random variables are computed
        from the joint distribution of the space,
        which is rebuilt from the settings.

        Reading a bound hands out a frozen copy of the corresponding array
        of the distribution, which the variable leaves untouched.
    """

    distribution_settings: tuple[BaseDistributionSettings, ...] = Field(
        min_length=1,
        description=(
            "The settings of the marginal probability distributions of the components."
        ),
    )

    @field_validator(_distribution_settings_tag)
    @classmethod
    def __check_libraries(
        cls, settings: tuple[BaseDistributionSettings, ...]
    ) -> tuple[BaseDistributionSettings, ...]:
        """Check that the marginal probability distributions share a single library.

        Args:
            settings: The settings of the marginal probability distributions.

        Returns:
            The settings of the marginal probability distributions.
        """
        _check_distribution_libraries({setting._library_name for setting in settings})
        return settings

    @cached_property
    def __cached_distribution(self) -> BaseJointDistribution:
        """The joint probability distribution of the components, built on demand."""
        settings = self.distribution_settings
        marginal_class = distribution_factory.get_class(settings[0].target_class_name)
        joint_class = marginal_class.joint_distribution_class
        return joint_class(joint_class.settings_class(marginal_settings=settings))

    @property
    def distribution(self) -> BaseJointDistribution:
        """The joint probability distribution of the components."""
        # The cache is hidden behind this property because pydantic lets a
        # functools.cached_property be overwritten, even on a frozen model.
        return self.__cached_distribution

    @property
    def size(self) -> int:
        """The size of the variable."""
        return len(self.distribution_settings)

    @property
    def type(self) -> DataType:
        """The type of data."""
        return DataType.FLOAT

    @property
    def lower_bound(self) -> RealArray:
        """The lower bound of the variable (read-only)."""
        # The bound arrays belong to the distribution, which is shared with every
        # copy of the variable, so hand out frozen copies and leave them untouched.
        return freeze_array(self.distribution.math_lower_bound)

    @property
    def upper_bound(self) -> RealArray:
        """The upper bound of the variable (read-only)."""
        return freeze_array(self.distribution.math_upper_bound)

    def filter_components(self, components: Sequence[int]) -> Self:  # noqa: D102
        settings = self.distribution_settings
        if list(components) == list(range(len(settings))):
            # Keeping every component in order is an identity;
            # the frozen variable can be shared.
            return self

        return type(self)(
            distribution_settings=tuple(settings[index] for index in components)
        )
