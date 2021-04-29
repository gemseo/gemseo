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
"""Test quality measure module."""
from __future__ import absolute_import, division, unicode_literals

import pytest

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure


def test_constructor():
    """Test construction."""
    dataset = Dataset()
    algo = MLAlgo(dataset)
    measure = MLQualityMeasure(algo)
    assert measure.algo is not None
    assert measure.algo.learning_set is dataset


def test_evaluate():
    """Test evaluation of quality measure."""
    dataset = Dataset()
    test_dataset = Dataset()
    algo = MLAlgo(dataset)
    measure = MLQualityMeasure(algo)
    with pytest.raises(NotImplementedError):
        measure.evaluate()
    with pytest.raises(NotImplementedError):
        measure.evaluate(MLQualityMeasure.LEARN)
    with pytest.raises(NotImplementedError):
        measure.evaluate(MLQualityMeasure.TEST, test_data=test_dataset)
    with pytest.raises(NotImplementedError):
        measure.evaluate(MLQualityMeasure.LOO)
    with pytest.raises(NotImplementedError):
        measure.evaluate(MLQualityMeasure.KFOLDS, n_folds=5)
    with pytest.raises(NotImplementedError):
        measure.evaluate(MLQualityMeasure.BOOTSTRAP, n_replicates=100)
