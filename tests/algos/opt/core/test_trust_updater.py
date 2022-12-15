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

from unittest import TestCase

from gemseo.algos.opt.core.trust_updater import BoundsUpdater
from gemseo.algos.opt.core.trust_updater import PenaltyUpdater
from gemseo.algos.opt.core.trust_updater import RadiusUpdater
from gemseo.algos.opt.core.trust_updater import TrustUpdater
from numpy import allclose
from numpy import ones
from numpy import zeros


class TestTrustUpdater(TestCase):
    """A class to test the TrustUpdater class."""

    def test_not_implemented_errors(self):
        trust_updater = TrustUpdater(
            thresholds=(0.0, 0.25), multipliers=(0.5, 2.0), bound=1e-6
        )
        self.assertRaises(NotImplementedError, trust_updater._check)
        self.assertRaises(NotImplementedError, trust_updater.update, 1.0, 1.0)


class TestPenaltyUpdater(TestCase):
    """A class to test the PenaltyUpdater class."""

    def test_invalid_parameters(self):
        """Tests the invalid parameters exceptions."""
        self.assertRaises(Exception, PenaltyUpdater, thresholds=0.1)
        self.assertRaises(Exception, PenaltyUpdater, thresholds=(0.1,))
        self.assertRaises(Exception, PenaltyUpdater, thresholds=(0.2, 0.1))
        self.assertRaises(Exception, PenaltyUpdater, multipliers=1.0)
        self.assertRaises(Exception, PenaltyUpdater, multipliers=(1.0,))
        self.assertRaises(Exception, PenaltyUpdater, multipliers=(2.0, 1.0))
        self.assertRaises(Exception, PenaltyUpdater, multipliers=(0.5, 0.5))
        PenaltyUpdater(multipliers=(0.5, 2.0))

    def test_failure(self):
        """Tests the failure case of the evaluate method."""
        updater = PenaltyUpdater(
            thresholds=(1.0, 2.0), multipliers=(0.5, 2.0), bound=1e-10
        )
        # Non-zero penalty parameter:
        new_penalty, success = updater.update(0.5, 1.0)
        self.assertAlmostEqual(new_penalty, 2.0, places=15)
        assert not success
        # Zero penalty parameter:
        new_penalty, success = updater.update(0.5, 0.0)
        self.assertAlmostEqual(new_penalty, 1e-10, places=15)
        assert not success

    def test_success_and_contraction(self):
        """Tests the success&contract case of the update method."""
        updater = PenaltyUpdater(
            thresholds=(1.0, 2.0), multipliers=(0.5, 2.0), bound=1e-10
        )
        # Non-zero penalty parameter:
        new_penalty, success = updater.update(2.5, 1.0)
        self.assertAlmostEqual(new_penalty, 0.5, places=15)
        assert success
        # Zero penalty parameter:
        new_penalty, success = updater.update(2.5, 0.0)
        self.assertAlmostEqual(new_penalty, 0.0, places=15)
        assert success

    def test_success_and_expansion(self):
        """Tests the success&expand case of the update method."""
        updater = PenaltyUpdater(
            thresholds=(1.0, 2.0), multipliers=(0.5, 2.0), bound=1e-10
        )
        # Non-zero penalty parameter:
        new_penalty, success = updater.update(1.5, 1.0)
        self.assertAlmostEqual(new_penalty, 2.0, places=15)
        assert success
        # Zero penalty parameter:
        new_penalty, success = updater.update(1.5, 0.0)
        self.assertAlmostEqual(new_penalty, 1e-10, places=15)
        assert success


class TestRadiusUpdater(TestCase):
    """A class to test the RadiusUpdater class."""

    def test_invalid_parameters(self):
        """Tests the invalid parameters exceptions."""
        self.assertRaises(Exception, RadiusUpdater, thresholds=0.1)
        self.assertRaises(Exception, RadiusUpdater, thresholds=(0.1,))
        self.assertRaises(Exception, RadiusUpdater, thresholds=(0.2, 0.1))
        self.assertRaises(Exception, RadiusUpdater, multipliers=1.0)
        self.assertRaises(Exception, RadiusUpdater, multipliers=(1.0,))
        self.assertRaises(Exception, RadiusUpdater, multipliers=(2.0, 1.0))
        self.assertRaises(Exception, RadiusUpdater, multipliers=(0.5, 0.5))
        RadiusUpdater(multipliers=(0.5, 2.0))

    def test_failure(self):
        """Tests the failure case of the evaluate method."""
        updater = RadiusUpdater(
            thresholds=(1.0, 2.0), multipliers=(0.5, 2.0), bound=10.0
        )
        new_radius, success = updater.update(0.5, 1.0)
        self.assertAlmostEqual(new_radius, 0.5, places=15)
        assert not success

    def test_success_and_contraction(self):
        """Tests the success&contract case of the update method."""
        updater = RadiusUpdater(
            thresholds=(1.0, 2.0), multipliers=(0.5, 2.0), bound=10.0
        )
        new_radius, success = updater.update(1.5, 1.0)
        self.assertAlmostEqual(new_radius, 0.5, places=15)
        assert success

    def test_success_and_expansion(self):
        """Tests the success&expand case of the update method."""
        updater = RadiusUpdater(
            thresholds=(1.0, 2.0), multipliers=(0.5, 2.0), bound=10.0
        )
        # Non-maximal penalty parameter:
        new_radius, success = updater.update(2.5, 1.0)
        self.assertAlmostEqual(new_radius, 2.0, places=15)
        assert success
        # Maximal penalty parameter:
        new_radius, success = updater.update(2.5, 10.0)
        self.assertAlmostEqual(new_radius, 10.0, places=15)
        assert success


class TestBoundsUpdater(TestCase):
    """A class to test the BoundsUpdater class."""

    def setUp(self):
        dim = 5
        self.dim = dim
        self.upper_bounds = 5.0 * ones(dim)
        self.lower_bounds = -3.0 * ones(dim)
        self.center = 2.0 * ones(dim)

    def test_nonnormalized_update(self):
        """Tests the non-normalized update of trust bounds."""
        trust_bounds = BoundsUpdater(self.lower_bounds, self.upper_bounds)
        lower_bounds, upper_bounds = trust_bounds.update(0.5, self.center)
        assert allclose(lower_bounds, 1.5 * ones(self.dim))
        assert allclose(upper_bounds, 2.5 * ones(self.dim))

    def test_normalized_update(self):
        """Tests the normalized update of trust bounds."""
        trust_bounds = BoundsUpdater(
            self.lower_bounds, self.upper_bounds, normalize=True
        )

        lower_bounds, upper_bounds = trust_bounds.update(0.5, self.center)
        assert allclose(lower_bounds, zeros(self.dim))
        assert allclose(upper_bounds, 4.0 * ones(self.dim))
