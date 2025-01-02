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

from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from gemseo.mlearning.regression.algos.rbf_settings import RBF
from gemseo.mlearning.regression.algos.thin_plate_spline import TPSRegressor


def test_init(dataset):
    """Check the default initialization of a TPSRegressor."""
    algo = TPSRegressor(dataset)
    algo.learn()
    assert isinstance(algo, RBFRegressor)
    assert algo.algo.function == RBF.THIN_PLATE
    assert algo.algo.smooth == 0.0
    assert algo.algo.norm == "euclidean"


def test_init_custom(dataset):
    """Check the custom initialization of a TPSRegressor."""
    algo = TPSRegressor(dataset, norm="minkowski", smooth=0.1)
    algo.learn()
    assert algo.algo.smooth == 0.1
    assert algo.algo.norm == "minkowski"
