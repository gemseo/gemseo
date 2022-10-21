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
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.uncertainty.distributions.distribution import Distribution


def test_raise_notimplementederror():
    dist = Distribution("x", "Normal", {"mu": 1, "sigma": 0}, 1)
    with pytest.raises(NotImplementedError):
        dist.compute_samples()
    with pytest.raises(NotImplementedError):
        dist.compute_cdf(1)
    with pytest.raises(NotImplementedError):
        dist.compute_inverse_cdf(1)
    with pytest.raises(NotImplementedError):
        dist.mean
    with pytest.raises(NotImplementedError):
        dist.standard_deviation
    with pytest.raises(NotImplementedError):
        pdf = dist._pdf(1)
        pdf(1)
    with pytest.raises(NotImplementedError):
        cdf = dist._cdf(1)
        cdf(1)
