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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Draw surfaces from a [Dataset][gemseo.datasets.dataset.Dataset].

A [Surfaces][gemseo.post.dataset.surfaces.Surfaces] plot represents samples
of a functional variable $z(x,y)$ discretized over a 2D mesh.
Both evaluations of $z$ and mesh are stored
in a [Dataset][gemseo.datasets.dataset.Dataset],
$z$ as a parameter and the mesh as a metadata.
"""

from __future__ import annotations

from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.surfaces_settings import Surfaces_Settings


class Surfaces(BaseDatasetPlot[Surfaces_Settings]):
    """Plot surfaces y_i over the mesh x."""

    settings_class = Surfaces_Settings
