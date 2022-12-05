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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Benoît Pauwels - refactoring
"""Updates a trust parameter according to a decreases' ratio."""
from __future__ import annotations

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import divide
from numpy import maximum
from numpy import minimum
from numpy import multiply
from numpy import ndarray


class TrustUpdater(metaclass=GoogleDocstringInheritanceMeta):
    """Updater of a trust parameter."""

    def __init__(
        self,
        thresholds: tuple[float, float],
        multipliers: tuple[float, float],
        bound=None,
    ) -> None:
        """
        Args:
            thresholds: The thresholds for the decreases' ratio.
            multipliers: The multipliers for the trust parameter.
            bound: The absolute bound for the trust parameter.
        """  # noqa: D205, D212, D415
        if not isinstance(thresholds, tuple):
            raise ValueError(
                "The thresholds must be input as a tuple; "
                f"input of type {type(thresholds)} was provided."
            )
        self._ratio_thresholds = thresholds
        if not isinstance(multipliers, tuple):
            raise ValueError(
                "The multipliers must be input as a tuple; "
                f"input of type {type(multipliers)} was provided."
            )
        self._param_multipliers = multipliers
        self._param_bound = bound

    def _check(self) -> None:
        """Check the consistency of the attributes."""
        raise NotImplementedError()

    def update(self, ratio: float, parameter: float) -> tuple[float, bool]:
        """Update the trust parameter relative to the decrease ratio value.

        Method to be overridden by subclasses.

        Args:
            ratio: The decrease ratio.
            parameter: The trust parameter (radius or penalty).

        Returns:
            The new trust parameter, the iteration success.
        """
        raise NotImplementedError()


class PenaltyUpdater(TrustUpdater):
    """Update the penalty parameter."""

    def __init__(  # noqa:D107
        self,
        thresholds: tuple[float, float] = (0.0, 0.25),
        multipliers: tuple[float, float] = (0.5, 2.0),
        bound: float = 1e-6,
    ) -> None:
        super().__init__(thresholds, multipliers, bound)
        self._check()

    def _check(self) -> None:
        # Check the thresholds:
        if len(self._ratio_thresholds) != 2:
            raise ValueError(
                "There must be exactly two thresholds for the "
                f"decreases ratio; {len(self._ratio_thresholds)} were given."
            )
        update_thresh = self._ratio_thresholds[0]
        nonexp_thresh = self._ratio_thresholds[1]
        if update_thresh > nonexp_thresh:
            raise ValueError(
                f"The update threshold ({update_thresh}) must be lower than or equal "
                f"to the non-expansion threshold ({nonexp_thresh})."
            )
        # Check the multipliers:
        if len(self._param_multipliers) != 2:
            raise ValueError(
                "There must be exactly two multipliers for the "
                f"penalty parameter; {len(self._ratio_thresholds)} were given."
            )
        contract_fact = self._param_multipliers[0]
        expan_fact = self._param_multipliers[1]
        if contract_fact >= 1.0:
            raise ValueError(
                f"The contraction factor ({contract_fact}) must be lower than one."
            )
        if expan_fact < 1.0:
            raise ValueError(
                f"The expansion factor ({expan_fact}) "
                "must be greater than or equal to one."
            )

    def update(self, ratio: float, parameter: float) -> tuple[float, bool]:  # noqa:D102
        # The iteration is declared successful if and only if the ratio is
        #         greater than or equal to the lower threshold.
        success = ratio >= self._ratio_thresholds[0]
        # If the ratio is greater than or equal to the upper threshold, the
        # penalty parameter is not increased, otherwise it is increased.
        if ratio >= self._ratio_thresholds[1]:
            new_penalty = parameter * self._param_multipliers[0]
            if self._param_bound is not None and new_penalty < self._param_bound:
                new_penalty = 0.0
        else:
            if self._param_bound is not None and parameter == 0.0:
                new_penalty = self._param_bound
            else:
                new_penalty = parameter * self._param_multipliers[1]
        return new_penalty, success


class RadiusUpdater(TrustUpdater):
    """Update the radius of the trust region."""

    def __init__(  # noqa:D107
        self,
        thresholds: tuple[float, float] = (0.0, 0.25),
        multipliers: tuple[float, float] = (0.5, 2.0),
        bound: float = 1000.0,
    ) -> None:
        super().__init__(thresholds, multipliers, bound)
        self._check()

    def _check(self) -> None:
        # Check the thresholds:
        if len(self._ratio_thresholds) != 2:
            raise ValueError(
                "There must be exactly two thresholds for the "
                f"decreases ratio; {len(self._ratio_thresholds)} were given."
            )
        update_thresh = self._ratio_thresholds[0]
        noncontract_thresh = self._ratio_thresholds[1]
        if update_thresh > noncontract_thresh:
            raise ValueError(
                f"The update threshold ({update_thresh}) must be lower than or equal "
                f"to the non-contraction threshold ({noncontract_thresh})."
            )
        # Check the multipliers:
        if len(self._param_multipliers) != 2:
            raise ValueError(
                "There must be exactly two multipliers for the region radius; "
                f"{len(self._ratio_thresholds)} were given."
            )
        contract_fact = self._param_multipliers[0]
        expan_fact = self._param_multipliers[1]
        if contract_fact > 1.0:
            raise ValueError(
                f"The contraction factor ({contract_fact}) "
                f"must be lower than or equal to one."
            )
        if expan_fact <= 1.0:
            raise ValueError(
                f"The expansion factor ({expan_fact}) must be greater than one."
            )

    def update(self, ratio: float, parameter: float) -> tuple[float, bool]:  # noqa:D102
        # The iteration is declared successful if and only if the ratio is
        # greater than or equal to the lower threshold.
        success = ratio >= self._ratio_thresholds[0]
        # If the ratio is greater than or equal to the upper threshold, the
        # region radius is not decreased, otherwise it is decreased.
        if ratio >= self._ratio_thresholds[1]:
            new_radius = parameter * self._param_multipliers[1]
            if self._param_bound is not None:
                new_radius = min(new_radius, self._param_bound)
        else:
            new_radius = parameter * self._param_multipliers[0]
        return new_radius, success


class BoundsUpdater:
    """Updater of the trust bounds, i.e. trust ball w.r.t.

    the infinity norm.
    """

    def __init__(
        self, lower_bounds: ndarray, upper_bounds: ndarray, normalize: bool = False
    ) -> None:
        """
        Args:
            lower_bounds: The reference lower bounds.
            upper_bounds: The reference upper bounds.
            normalize: Whether to apply the radius to the normalized bounds.
        """  # noqa: D205, D212, D415
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self._normalized_update = normalize
        self.__bound_sum = lower_bounds + upper_bounds
        self.__bound_diff = upper_bounds - lower_bounds

    def update(self, radius: float, center: ndarray) -> tuple[ndarray, ndarray]:
        """Update the trust bounds.

        Args:
            radius: The region radius w.r.t. the infinity norm.
            center: The center of the region

        Returns:
            The updated lower and upper bounds of the trust region.
        """
        if self._normalized_update:
            radius = radius * 0.5 * self.__bound_diff

        return self._compute_trust_bounds(
            self._lower_bounds, self._upper_bounds, center, radius
        )

    @staticmethod
    def _compute_trust_bounds(
        lower_bounds: ndarray,
        upper_bounds: ndarray,
        center: ndarray,
        radius: float | ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Update the bounds of the trust region.

        Use a ball center and ball radius w.r.t. the infinity norm.

        Args:
            lower_bounds: The lower bounds to be updated.
            upper_bounds: The upper bounds to be updated.
            center: The center of the ball.
            radius: The radius of the ball;
                either the same for all coordinate or coordinate-specific.

        Returns:
            The updated lower and upper bounds.
        """
        return (
            minimum(maximum(center - radius, lower_bounds), upper_bounds),
            maximum(minimum(center + radius, upper_bounds), lower_bounds),
        )

    def _normalize(self, x_vect: ndarray) -> ndarray:
        """Normalize the coordinates of a vector to [-1, 1].

        Args:
            x_vect: The vector to normalize.

        Returns:
            The normalized vector.
        """
        return divide(2.0 * x_vect - self.__bound_sum, self.__bound_diff)

    def _unnormalize(self, x_norm: ndarray) -> ndarray:
        """Unnormalize the coordinates of a vector to [-1, 1].

        Args:
            x_norm: The vector to unnormalize.

        Returns:
            The unnormalized vector.
        """
        return (multiply(x_norm, self.__bound_diff) + self.__bound_sum) / 2.0
