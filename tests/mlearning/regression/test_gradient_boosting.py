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
from numpy import array
from sklearn.ensemble import GradientBoostingRegressor as SKLGradientBoosting

from gemseo.mlearning.regression.algos.gradient_boosting import (
    GradientBoostingRegressor,
)


def test_init(dataset):
    """Check that the wrapped algorithm is GradientBoostingRegressor from sklearn.

    Check also the default number of estimators.
    """
    for algo in GradientBoostingRegressor(dataset).algo:
        assert isinstance(algo, SKLGradientBoosting)
        assert algo.n_estimators == 100


def test_init_n_estimators(dataset):
    """Check that the number of estimators can be changed."""
    for algo in GradientBoostingRegressor(dataset, n_estimators=10).algo:
        assert algo.n_estimators == 10


def test_fit(dataset, input_data, output_data):
    """Check the learning stage."""
    gdr = GradientBoostingRegressor(dataset)
    for algo in gdr.algo:
        assert not hasattr(algo, "n_estimators_")
    gdr._fit(input_data, output_data)
    for algo in gdr.algo:
        assert algo.n_estimators_ == 100


def test_predict(dataset):
    """Check the prediction stage."""
    algo = GradientBoostingRegressor(dataset)
    algo.learn()
    input_data = array([[1.0, 1.0]])
    assert algo._predict(input_data).shape == (1, 1)

    with pytest.raises(NotImplementedError):
        algo.predict_jacobian(input_data)
