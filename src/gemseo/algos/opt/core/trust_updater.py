# -*- coding: utf-8 -*-
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
"""
Updates a trust parameter according to a decreases ratio
********************************************************
"""

from __future__ import division, unicode_literals

import logging

from numpy import divide, maximum, minimum, multiply

LOGGER = logging.getLogger(__name__)


class TrustUpdater(object):
    """Updates the trust parameter."""

    def __init__(self, thresholds=None, multipliers=None, bound=None):
        """Initializer.

        :param thresholds: thresholds for the decreases ratio
        :type thresholds: tuple
        :param multipliers: multipliers for the trust parameter
        :type multipliers: tuple
        :param bound: (lower or upper) bound for the trust parameter
        """
        if not isinstance(thresholds, tuple):
            raise ValueError(
                "The thresholds must be input as a tuple; "
                "input of type " + type(thresholds) + " was provided."
            )
        self._ratio_thresholds = thresholds
        if not isinstance(multipliers, tuple):
            raise ValueError(
                "The multipliers must be input as a tuple; "
                "input of type " + type(multipliers) + " was provided."
            )
        self._param_multipliers = multipliers
        self._param_bound = bound  # bound for the trust parameter

    def _check(self):
        """Checks attributes consistency."""
        raise NotImplementedError()

    def update(self, ratio, parameter):
        """Updates the trust parameter relative to the decreases ratio value. Method to
        be overidden by subclasses.

        :param ratio: decreases ratio
        :param parameter: trust parameter (radius or penalty)
        :returns: new trust parameter, iteration success
        :rtype: bool
        """
        raise NotImplementedError()


class PenaltyUpdater(TrustUpdater):
    """Updates the penalty parameter."""

    def __init__(self, thresholds=(0.0, 0.25), multipliers=(0.5, 2.0), bound=1e-6):
        """Initializer.

        :param thresholds: thresholds for the decreases ratio
        :type thresholds: tuple
        :param multipliers: multipliers for the penalty parameter
        :type multipliers: tuple
        :param bound: lower bound for the penalty parameter
        """
        super(PenaltyUpdater, self).__init__(thresholds, multipliers, bound)
        self._check()

    def _check(self):
        """Checks the attributes of the penalty updater."""
        # Check the thresholds:
        if len(self._ratio_thresholds) != 2:
            raise ValueError(
                "There must be exactly two thresholds for the "
                "decreases ratio; " + str(len(self._ratio_thresholds)) + " were given."
            )
        update_thresh = self._ratio_thresholds[0]
        nonexp_thresh = self._ratio_thresholds[1]
        if update_thresh > nonexp_thresh:
            raise ValueError(
                "The update threshold ("
                + str(update_thresh)
                + ") must be lower than or equal to "
                + "the non-expansion threshold ("
                + str(nonexp_thresh)
                + ")."
            )
        # Check the multipliers:
        if len(self._param_multipliers) != 2:
            raise ValueError(
                "There must be exactly two multipliers for the "
                "penalty parameter; "
                + str(len(self._ratio_thresholds))
                + " were given."
            )
        contract_fact = self._param_multipliers[0]
        expan_fact = self._param_multipliers[1]
        if contract_fact >= 1.0:
            raise ValueError(
                "The contraction factor ("
                + str(contract_fact)
                + ") must be lower than one."
            )
        if expan_fact < 1.0:
            raise ValueError(
                "The expansion factor ("
                + str(expan_fact)
                + ") must be greater than or equal to one."
            )

    def update(self, ratio, parameter):
        """Updates the penalty parameter.

        :param ratio: decreases ratio
        :param parameter: penalty parameter
        :returns: new penalty parameter, iteration success (boolean)
        """
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
    """Updates the trust region radius."""

    def __init__(self, thresholds=(0.0, 0.25), multipliers=(0.5, 2.0), bound=1000.0):
        """Initializer.

        :param thresholds: thresholds for the decreases ratio
        :type thresholds: tuple
        :param multipliers: multipliers for the region radius
        :type multipliers: tuple
        :param bound: lower bound for the region radius
        """
        super(RadiusUpdater, self).__init__(thresholds, multipliers, bound)
        self._check()

    def _check(self):
        """Checks the attributes of the radius updater."""
        # Check the thresholds:
        if len(self._ratio_thresholds) != 2:
            raise ValueError(
                "There must be exactly two thresholds for the "
                "decreases ratio; " + str(len(self._ratio_thresholds)) + " were given."
            )
        update_thresh = self._ratio_thresholds[0]
        noncontract_thresh = self._ratio_thresholds[1]
        if update_thresh > noncontract_thresh:
            raise ValueError(
                "The update threshold ("
                + str(update_thresh)
                + ") must be lower than or equal to the "
                "non-contraction threshold (" + str(noncontract_thresh) + ")."
            )
        # Check the multipliers:
        if len(self._param_multipliers) != 2:
            raise ValueError(
                "There must be exactly two multipliers for the "
                "region radius; " + str(len(self._ratio_thresholds)) + " were  given."
            )
        contract_fact = self._param_multipliers[0]
        expan_fact = self._param_multipliers[1]
        if contract_fact > 1.0:
            raise ValueError(
                "The contraction factor ("
                + str(contract_fact)
                + ") must be lower than or equal to one."
            )
        if expan_fact <= 1.0:
            raise ValueError(
                "The expansion factor ("
                + str(expan_fact)
                + ") must be greater than one."
            )

    def update(self, ratio, parameter):
        """Updates the trust radius.

        :param ratio: decreases ratio
        :param parameter: region radius
        :returns: new region radius, iteration success (boolean)
        """
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


class BoundsUpdater(object):
    """Updates trust bounds, i.e. trust ball w.r.t.

    the infinity norm.
    """

    def __init__(self, lower_bounds, upper_bounds, normalize=False):
        """Initializer.

        :param lower_bounds: reference lower bounds
        :type lower_bounds: ndarray
        :param upper_bounds: reference upper bounds
        :type upper_bounds: ndarray
        :param normalize: if True the radius is applied to the normalized
            bounds
        :param normalize: bool, optional
        """
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self._normalized_update = normalize

    def update(self, radius, center):
        """Updates the trust bounds.

        :param radius: region radius w.r.t. the infinity norm
        :type radius: float
        :param center: region center
        :type center: ndarray
        :returns: new region radius, iteration success (boolean)
        """
        if not self._normalized_update:
            low_bnds, upp_bnds = self._compute_trust_bounds(
                self._lower_bounds, self._upper_bounds, center, radius
            )
        else:
            # Compute the normalized coordinate-specific radii:
            # radius_i = radius * 0.5 * (upper_bound_i - lower_bound_i)
            radii = radius * 0.5 * (self._upper_bounds - self._lower_bounds)
            # Compute the trust bounds
            low_bnds, upp_bnds = self._compute_trust_bounds(
                self._lower_bounds, self._upper_bounds, center, radii
            )

        return low_bnds, upp_bnds

    @staticmethod
    def _compute_trust_bounds(lower_bounds, upper_bounds, center, radius):
        """Updates bounds based on a ball center and ball radius w.r.t. the infinity
        norm.

        :param lower_bounds: lower bounds to be updated
        :type lower_bounds: ndarray
        :param upper_bounds: upper bounds to be updated
        :type upper_bounds: ndarray
        :param center: ball center
        :type center: ndarray
        :param radius: ball radius (same for all coordinate or coordinate-specific)
        :type radius: float or ndarray
        :returns: updated lower bounds, updated upper bounds
        :rtype: ndarray, ndarray
        """
        lower_trust_bounds = minimum(
            maximum(center - radius, lower_bounds), upper_bounds
        )
        upper_trust_bounds = maximum(
            minimum(center + radius, upper_bounds), lower_bounds
        )
        return lower_trust_bounds, upper_trust_bounds

    def _normalize(self, x_vect):
        """Normalize a vector coordinates to [-1, 1]."""
        x_norm = 2.0 * x_vect - self._upper_bounds - self._lower_bounds
        x_norm = divide(x_norm, self._upper_bounds - self._lower_bounds)
        return x_norm

    def _unnormalize(self, x_norm):
        """Unnormalize a vector coordinates from [-1, 1]."""
        x_vect = multiply(x_norm, (self._upper_bounds - self._lower_bounds))
        x_vect = (x_vect + self._upper_bounds + self._lower_bounds) / 2.0
        return x_vect
