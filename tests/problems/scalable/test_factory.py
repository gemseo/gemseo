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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import array

from gemseo.caches.memory_full_cache import MemoryFullCache
from gemseo.problems.scalable.factory import ScalableModelFactory

standard_library.install_aliases()


def test_constructor():
    ScalableModelFactory()


def test_create():
    factory = ScalableModelFactory()
    cache = MemoryFullCache()
    input_names = ["x", "y"]
    output_names = ["z"]
    for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        data = {"x": array([val] * 2), "y": array([val]), "z": array([val])}
        cache.cache_outputs(data, input_names, data, output_names)
    cache.cache_outputs(data, input_names, data, output_names)
    factory.create("ScalableDiagonalModel", data=cache)


def test_list_available():
    factory = ScalableModelFactory()
    assert "ScalableDiagonalModel" in factory.scalable_models


def test_is_available():
    factory = ScalableModelFactory()
    assert factory.is_available("ScalableDiagonalModel")
