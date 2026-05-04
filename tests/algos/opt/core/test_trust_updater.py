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
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests TrustUpdater."""

from __future__ import annotations

import pytest
from numpy import full
from numpy.testing import assert_allclose

from gemseo.algos.opt.core.trust_updater import BoundsUpdater
from gemseo.algos.opt.core.trust_updater import PenaltyUpdater
from gemseo.algos.opt.core.trust_updater import RadiusUpdater
from gemseo.utils.testing.helpers import assert_exception


@pytest.mark.parametrize(
    ("thresholds", "type_"),
    [
        (0.1, TypeError),
        ((0.1,), ValueError),
        ((0.2, 0.1), ValueError),
    ],
)
@pytest.mark.parametrize("cls", [RadiusUpdater, PenaltyUpdater])
def test_invalid_thresholds(thresholds, type_, cls, snapshot) -> None:
    """Tests the invalid thresholds exceptions."""
    with assert_exception(type_, snapshot):
        cls(thresholds=thresholds)


@pytest.mark.parametrize(
    ("multipliers", "type_"),
    [
        (1.0, TypeError),
        ((1.0,), ValueError),
        ((2.0, 1.0), ValueError),
        ((0.5, 0.5), ValueError),
    ],
)
@pytest.mark.parametrize("cls", [RadiusUpdater, PenaltyUpdater])
def test_invalid_multipliers(multipliers, type_, cls, snapshot) -> None:
    """Tests the invalid multipliers exceptions."""
    with assert_exception(type_, snapshot):
        cls(multipliers=multipliers)


@pytest.mark.parametrize(
    ("ratio", "parameter", "expected_new_penalty", "expected_success"),
    [
        (0.5, 1.0, 2.0, False),
        (0.5, 0.0, 1e-10, False),
        (1.5, 1.0, 2.0, True),
        (1.5, 0.0, 1e-10, True),
        (2.5, 1.0, 0.5, True),
        (2.5, 0.0, 0.0, True),
    ],
)
def test_penalty_updater(
    ratio, parameter, expected_new_penalty, expected_success
) -> None:
    """Tests the update method of PenaltyUpdater."""
    updater = PenaltyUpdater(thresholds=(1.0, 2.0), multipliers=(0.5, 2.0), bound=1e-10)
    new_penalty, success = updater.update(ratio, parameter)
    assert new_penalty == pytest.approx(expected_new_penalty)
    assert success is expected_success


@pytest.mark.parametrize(
    ("ratio", "parameter", "expected_new_radius", "expected_success"),
    [
        (0.5, 1.0, 0.5, False),
        (1.5, 1.0, 0.5, True),
        (2.5, 1.0, 2.0, True),
        (2.5, 10.0, 10.0, True),
    ],
)
def test_radius_updater(
    ratio, parameter, expected_new_radius, expected_success
) -> None:
    """Tests the update method of RadiusUpdater."""
    updater = RadiusUpdater(thresholds=(1.0, 2.0), multipliers=(0.5, 2.0), bound=10.0)
    new_radius, success = updater.update(ratio, parameter)
    assert new_radius == pytest.approx(expected_new_radius)
    assert success is expected_success


@pytest.mark.parametrize(
    ("kwargs", "expected_lower_bounds", "expected_upper_bounds"),
    [
        ({}, full(5, 1.5), full(5, 2.5)),
        ({"normalize": True}, full(5, 0.0), full(5, 4.0)),
    ],
)
def test_bounds_updater(kwargs, expected_lower_bounds, expected_upper_bounds) -> None:
    """Tests the update method of BoundsUpdater."""
    trust_bounds = BoundsUpdater(full(5, -3.0), full(5, 5.0), **kwargs)
    lower_bounds, upper_bounds = trust_bounds.update(0.5, full(5, 2.0))
    assert_allclose(lower_bounds, expected_lower_bounds)
    assert_allclose(upper_bounds, expected_upper_bounds)
