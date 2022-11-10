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
from __future__ import annotations

from gemseo.uncertainty.use_cases.ishigami.ishigami_space import IshigamiSpace
from numpy import pi


def test_ishigami_space():
    """Check the Ishigami space."""
    space = IshigamiSpace()
    assert space.dimension == 3
    assert space.variables_names == ["x1", "x2", "x3"]
    for distribution in space.distributions.values():
        assert len(distribution.marginals) == 1
        distribution = distribution.marginals[0]
        assert distribution.kwds == {"loc": -pi, "scale": 2 * pi}
        assert distribution.dist.__class__.__name__ == "uniform_gen"
