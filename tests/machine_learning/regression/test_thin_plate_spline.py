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

from gemseo.machine_learning.regression.models.rbf import RBFRegressor
from gemseo.machine_learning.regression.models.rbf_settings import RBF
from gemseo.machine_learning.regression.models.thin_plate_spline import TPSRegressor
from gemseo.machine_learning.regression.models.thin_plate_spline_settings import (
    TPSRegressor_Settings,
)


def test_init(dataset):
    """Check the default initialization of a TPSRegressor."""
    model = TPSRegressor(dataset)
    model.learn()
    assert isinstance(model, RBFRegressor)
    assert model.algo.function == RBF.THIN_PLATE
    assert model.algo.smooth == 0.0
    assert model.algo.norm == "euclidean"


def test_init_custom(dataset):
    """Check the custom initialization of a TPSRegressor."""
    model = TPSRegressor(dataset, TPSRegressor_Settings(norm="minkowski", smooth=0.1))
    model.learn()
    assert model.algo.smooth == 0.1
    assert model.algo.norm == "minkowski"
