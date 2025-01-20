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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the empirical estimation of tolerance intervals."""

from __future__ import annotations

import pytest
from numpy.random import RandomState

from gemseo.datasets.dataset import Dataset
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    _BaseToleranceInterval,
)
from gemseo.uncertainty.statistics.tolerance_interval.empirical import (
    EmpiricalToleranceInterval,
)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """1000 samples of a Weibull probability distribution."""
    return Dataset.from_array(
        RandomState(0).weibull(1.5, size=1000),
        ["x_1"],
        {"x_1": 1},
    )


def test_empirical_quantile_both(dataset) -> None:
    """Check the bounds of an empirical two-sided TI."""
    tolerance_interval = EmpiricalToleranceInterval(
        dataset.get_view().to_numpy().ravel()
    )
    lower, upper = tolerance_interval.compute(0.95, 0.9)
    assert pytest.approx(lower, 0.01) == 0.0744460
    assert pytest.approx(upper, 0.01) == 2.4544127


def test_empirical_quantile_lower(dataset) -> None:
    """Check the bounds of an empirical lower-sided TI."""
    tolerance_interval = EmpiricalToleranceInterval(
        dataset.get_view().to_numpy().ravel()
    )
    lower, _ = tolerance_interval.compute(
        0.95,
        0.9,
        side=_BaseToleranceInterval.ToleranceIntervalSide.LOWER,  # noqa:E501
    )
    assert pytest.approx(lower, 0.01) == 0.130419


def test_empirical_quantile_upper(dataset) -> None:
    """Check the bounds of an empirical upper-sided TI."""
    tolerance_interval = EmpiricalToleranceInterval(
        dataset.get_view().to_numpy().ravel()
    )
    _, upper = tolerance_interval.compute(
        0.975, 0.9, side=_BaseToleranceInterval.ToleranceIntervalSide.UPPER
    )
    assert pytest.approx(upper, 0.01) == 2.47285
