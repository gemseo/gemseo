# -*- coding: utf-8 -*-
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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
""" Test error measure module. """
from __future__ import absolute_import, division, unicode_literals

import pytest
from future import standard_library

from gemseo.mlearning.qual_measure.error_measure import MLErrorMeasure
from gemseo.mlearning.regression.linreg import LinearRegression
from gemseo.problems.dataset.rosenbrock import RosenbrockDataset

standard_library.install_aliases()


@pytest.fixture
def measure():
    """ Error measure. """
    dataset = RosenbrockDataset(opt_naming=False)
    algo = LinearRegression(dataset)
    error_measure = MLErrorMeasure(algo)
    return error_measure


def test_evaluate(measure):
    """ Test different evaluation methods of error measure. """
    with pytest.raises(NotImplementedError):
        measure.evaluate_learn()
    dataset_test = RosenbrockDataset(opt_naming=False)
    with pytest.raises(NotImplementedError):
        measure.evaluate_test(dataset_test)
    with pytest.raises(NotImplementedError):
        measure.evaluate_loo()
    with pytest.raises(NotImplementedError):
        measure.evaluate_kfolds(n_folds=5)
    with pytest.raises(NotImplementedError):
        measure.evaluate_bootstrap(n_replicates=100)
