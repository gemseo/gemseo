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
from sklearn.svm import SVR

from gemseo.mlearning.regression.algos.svm import SVMRegressor


def test_init(dataset):
    """Check that the wrapped algorithm is SVR from sklearn.

    Check also the default number of estimators.
    """
    for algo in SVMRegressor(dataset).algo:
        assert isinstance(algo, SVR)
        assert algo.kernel == "rbf"


def test_init_kernel(dataset):
    """Check that the kernel can be changed."""
    for algo in SVMRegressor(dataset, kernel="linear").algo:
        assert algo.kernel == "linear"


def test_fit(dataset, input_data, output_data):
    """Check the learning stage."""
    svm = SVMRegressor(dataset)
    for algo in svm.algo:
        assert not hasattr(algo, "shape_fit_")
    svm._fit(input_data, output_data)
    for algo in svm.algo:
        assert algo.shape_fit_ == (100, 2)


def test_predict(dataset):
    """Check the prediction stage."""
    algo = SVMRegressor(dataset)
    algo.learn()
    input_data = array([[1.0, 1.0]])
    assert algo._predict(input_data).shape == (1, 1)

    with pytest.raises(NotImplementedError):
        algo.predict_jacobian(input_data)
