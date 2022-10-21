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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Reda El Amri
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.dirac import OTDiracDistribution
from numpy import array
from numpy.testing import assert_equal


def test_default():
    """Check the Dirac distribution with the default variable value."""
    distribution = OTDiracDistribution("x", dimension=3)
    assert_equal(
        distribution.compute_samples(2), array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )


def test_custom():
    """Check the Dirac distribution with a custom variable value."""
    distribution = OTDiracDistribution("x", variable_value=2.0, dimension=3)
    assert_equal(
        distribution.compute_samples(2), array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    )
