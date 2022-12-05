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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A k-means classification of the optimization history."""
from __future__ import annotations

import logging
from pathlib import Path

from numpy import array
from numpy import int as np_int
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from gemseo.post.opt_post_processor import OptPostProcessor

LOGGER = logging.getLogger(__name__)


class KMeans(OptPostProcessor):
    """Performs a k-means clustering on optimization history.

    The default number of clusters is 5 and can be modified in option.

    The k-means construction depends
    on the :code:`MiniBatchKMeans` class
    of the :code:`cluster` module of the
    `scikit-learn library <https://scikit-learn.org/stable/modules/generated/
    sklearn.cluster.MiniBatchKMeans.html>`_ .
    """

    def _run(
        self,
        save: bool = True,
        show: bool = False,
        file_path: Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_extension: str | None = None,
        fig_size: tuple[float, float] | None = None,
        n_clusters: int = 5,
    ) -> None:
        """
        Args:
            n_clusters: The number of clusters.
        """  # noqa: D205, D212, D415
        self.__build_clusters(n_clusters=n_clusters)

    def __build_clusters(
        self,
        n_clusters: int = 5,
    ) -> None:
        """Build the clusters.

        Args:
            n_clusters: The number of clusters.
        """
        x_history = self.database.get_x_history()
        x_vars = array(x_history)
        x_vars_sc = StandardScaler().fit_transform(x_vars)
        # estimate bandwidth for mean shift
        algorithm = cluster.MiniBatchKMeans(n_clusters=n_clusters)
        # predict cluster memberships
        algorithm.fit(x_vars_sc)
        y_pred = algorithm.labels_.astype(np_int)
        for x_vars, y_vars in zip(x_history, y_pred):
            self.database.store(x_vars, {"KM_cluster": int(y_vars)})
            self.materials_for_plotting[tuple(x_vars.real)] = y_vars
