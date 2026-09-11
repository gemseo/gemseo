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

from gemseo.problem.uncertainty.ishigami.statistics import mean
from gemseo.problem.uncertainty.ishigami.statistics import sobol_1
from gemseo.problem.uncertainty.ishigami.statistics import sobol_2
from gemseo.problem.uncertainty.ishigami.statistics import sobol_3
from gemseo.problem.uncertainty.ishigami.statistics import sobol_12
from gemseo.problem.uncertainty.ishigami.statistics import sobol_13
from gemseo.problem.uncertainty.ishigami.statistics import sobol_23
from gemseo.problem.uncertainty.ishigami.statistics import sobol_123
from gemseo.problem.uncertainty.ishigami.statistics import total_sobol_1
from gemseo.problem.uncertainty.ishigami.statistics import total_sobol_2
from gemseo.problem.uncertainty.ishigami.statistics import total_sobol_3
from gemseo.problem.uncertainty.ishigami.statistics import variance


@pytest.mark.parametrize(
    ("statistic", "value"),
    [
        (mean, 3.5),
        (variance, 13.84),
        (sobol_1, 0.31),
        (sobol_2, 0.44),
        (sobol_3, 0.0),
        (sobol_12, 0.0),
        (sobol_13, 0.24),
        (sobol_23, 0.0),
        (sobol_123, 0.0),
        (total_sobol_1, 0.55),
        (total_sobol_2, 0.44),
        (total_sobol_3, 0.24),
    ],
)
def test_statistics(statistic, value) -> None:
    assert statistic == pytest.approx(value, abs=0.01)
