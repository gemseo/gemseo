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

from gemseo.machine_learning.regression.model.rbf import RBFRegressor
from gemseo.machine_learning.regression.model.rbf_settings import RBF
from gemseo.machine_learning.regression.model.tps import TPSRegressor
from gemseo.machine_learning.regression.model.tps_settings import TPSRegressor_Settings


def test_init(dataset):
    """Check the default initialization of a TPSRegressor."""
    model = TPSRegressor(dataset)
    model.learn()
    assert isinstance(model, RBFRegressor)
    assert model.algo.kernel == RBF.THIN_PLATE_SPLINE
    assert (model.algo.smoothing == 0.0).all()


def test_init_custom(dataset):
    """Check the custom initialization of a TPSRegressor."""
    model = TPSRegressor(dataset, TPSRegressor_Settings(smoothing=0.1))
    model.learn()
    assert (model.algo.smoothing == 0.1).all()
