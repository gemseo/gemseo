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
"""Clusterers.

This package includes clustering models, a.k.a. clusterers.

Given an input data,
a clusterer is used to group data into classes, a.k.a. clusters.

Wherever possible,
these models should be able to predict the class of a new data,
as well as the probability of belonging to each class.

Use the
[ClustererFactory][gemseo.machine_learning.clustering.model.factory.ClustererFactory]
to access all the available clusterers
or derive either the
[BaseClusterer][gemseo.machine_learning.clustering.core.base_clusterer.BaseClusterer]
or
[BasePredictiveClusterer][gemseo.machine_learning.clustering.core.base_predictive_clusterer.BasePredictiveClusterer]
class
to add a new one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.machine_learning.clustering.model.factory import (
        CLUSTERER_FACTORY,  # noqa: F401
    )
    from gemseo.machine_learning.clustering.model.gaussian_mixture import (
        GaussianMixture,  # noqa: F401
    )
    from gemseo.machine_learning.clustering.model.kmeans import KMeans  # noqa: F401

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "CLUSTERER_FACTORY": "factory",
    "GaussianMixture": "gaussian_mixture",
    "KMeans": "kmeans",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
