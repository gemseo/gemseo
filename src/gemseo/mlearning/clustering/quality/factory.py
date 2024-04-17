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
"""A factory of objects to assess the quality of clusterers."""

from __future__ import annotations

from gemseo.mlearning.clustering.quality.base_clusterer_quality import (
    BaseClustererQuality,
)
from gemseo.mlearning.core.quality.factory import MLAlgoQualityFactory


class ClustererQualityFactory(MLAlgoQualityFactory):
    """A factory of objects to assess the quality of clusterers."""

    _CLASS = BaseClustererQuality
    _MODULE_NAMES = ("gemseo.mlearning.clustering.quality",)
