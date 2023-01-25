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
from gemseo.core.dataset import Dataset
from gemseo.mlearning.regression.gpr import GaussianProcessRegressor
from numpy import linspace
from numpy import newaxis
from numpy import pi
from numpy import sin


@pytest.fixture(scope="module")
def algo_for_transformer():
    """A GP regression of f(x) = x*sin(x)**2 over [0, 2*pi] with 20 points."""
    dataset = Dataset()
    x = linspace(0, 2 * pi, 20)[:, newaxis]
    dataset.add_variable("x", x, group="inputs")
    dataset.add_variable("y", x * sin(x) ** 2, group="outputs")
    algo = GaussianProcessRegressor(
        dataset,
        transformer=GaussianProcessRegressor.DEFAULT_TRANSFORMER,
        n_restarts_optimizer=0,
    )
    algo.learn()
    return algo
