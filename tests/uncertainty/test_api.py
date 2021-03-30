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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy.random import normal

from gemseo.core.dataset import Dataset
from gemseo.uncertainty.api import (
    create_distribution,
    create_statistics,
    get_available_distributions,
)
from gemseo.uncertainty.statistics.empirical import EmpiricalStatistics
from gemseo.uncertainty.statistics.parametric import ParametricStatistics

standard_library.install_aliases()


def test_available():
    distributions = get_available_distributions()
    assert "OTNormalDistribution" in distributions


def test_create():
    distribution = create_distribution("x", "OTNormalDistribution")
    assert distribution.mean[0] == 0.0


def test_create_statistics():
    n_samples = 100
    normal_rand = normal(size=n_samples).reshape((-1, 1))
    dataset = Dataset()
    dataset.set_from_array(normal_rand)
    stat = create_statistics(dataset)
    assert isinstance(stat, EmpiricalStatistics)
    stat = create_statistics(dataset, tested_distributions=["Normal", "Exponential"])
    assert isinstance(stat, ParametricStatistics)
