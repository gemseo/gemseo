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
"""OpenTURNS-based parametric estimation of statistics from a dataset."""

from __future__ import annotations

from typing import ClassVar

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_fitter import (
    OTDistributionFitter,
)
from gemseo.uncertainty.statistics.base_parametric import BaseParametricStatistics


class OTParametricStatistics(
    BaseParametricStatistics[
        OTDistribution,
        OTDistributionFitter.default_fitting_criterion,
        OTDistributionFitter.DistributionName,
        OTDistributionFitter.FittingCriterion,
        OTDistributionFitter.SignificanceTest,
    ]
):
    """A toolbox to compute statistics using OpenTURNS probability distribution-fitting."""  # noqa: E501

    DistributionName: ClassVar[OTDistributionFitter.DistributionName] = (
        OTDistributionFitter.DistributionName
    )
    FittingCriterion: ClassVar[OTDistributionFitter.FittingCriterion] = (
        OTDistributionFitter.FittingCriterion
    )
    SignificanceTest: ClassVar[OTDistributionFitter.SignificanceTest] = (
        OTDistributionFitter.SignificanceTest
    )
    _DISTRIBUTION_FITTER: ClassVar[OTDistributionFitter] = OTDistributionFitter
