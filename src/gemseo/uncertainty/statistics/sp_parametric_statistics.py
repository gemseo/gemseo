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
"""SciPy-based parametric estimation of statistics from a dataset."""

from __future__ import annotations

from typing import ClassVar

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.distribution_fitter import (
    SPDistributionFitter,
)
from gemseo.uncertainty.statistics.base_parametric_statistics import (
    BaseParametricStatistics,
)


class SPParametricStatistics(
    BaseParametricStatistics[
        SPDistribution,
        SPDistributionFitter.default_fitting_criterion,
        SPDistributionFitter.DistributionName,
        SPDistributionFitter.FittingCriterion,
        SPDistributionFitter.SignificanceTest,
    ]
):
    """A toolbox to compute statistics using SciPy probability distribution-fitting."""

    DistributionName: ClassVar[SPDistributionFitter.DistributionName] = (
        SPDistributionFitter.DistributionName
    )
    FittingCriterion: ClassVar[SPDistributionFitter.FittingCriterion] = (
        SPDistributionFitter.FittingCriterion
    )
    SignificanceTest: ClassVar[SPDistributionFitter.SignificanceTest] = (
        SPDistributionFitter.SignificanceTest
    )
    _DISTRIBUTION_FITTER: ClassVar[SPDistributionFitter] = SPDistributionFitter
