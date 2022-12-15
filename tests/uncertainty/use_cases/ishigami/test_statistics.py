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

import pytest
from gemseo.uncertainty.use_cases.ishigami.statistics import MEAN
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_1
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_12
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_123
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_13
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_2
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_23
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_3
from gemseo.uncertainty.use_cases.ishigami.statistics import TOTAL_SOBOL_1
from gemseo.uncertainty.use_cases.ishigami.statistics import TOTAL_SOBOL_2
from gemseo.uncertainty.use_cases.ishigami.statistics import TOTAL_SOBOL_3
from gemseo.uncertainty.use_cases.ishigami.statistics import VARIANCE


@pytest.mark.parametrize(
    "statistic,value",
    [
        (MEAN, 3.5),
        (VARIANCE, 13.84),
        (SOBOL_1, 0.31),
        (SOBOL_2, 0.44),
        (SOBOL_3, 0.0),
        (SOBOL_12, 0.0),
        (SOBOL_13, 0.24),
        (SOBOL_23, 0.0),
        (SOBOL_123, 0.0),
        (TOTAL_SOBOL_1, 0.55),
        (TOTAL_SOBOL_2, 0.44),
        (TOTAL_SOBOL_3, 0.24),
    ],
)
def test_statistics(statistic, value):
    assert statistic == pytest.approx(value, abs=0.01)
