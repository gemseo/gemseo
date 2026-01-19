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
#                         documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The base class for clustering models."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from numpy import ndarray
from numpy import unique

from gemseo.mlearning.clustering.models.base_clusterer_settings import (
    BaseClustererSettings,
)
from gemseo.mlearning.core.models.ml_model import (
    SavedObjectType as MLModelSavedObjectType,
)
from gemseo.mlearning.core.models.unsupervised import BaseMLUnsupervisedModel

if TYPE_CHECKING:
    from collections.abc import Sequence


SavedObjectType = MLModelSavedObjectType | ndarray | int


class BaseClusterer(BaseMLUnsupervisedModel):
    """The base class for clustering models."""

    labels: ndarray
    """The labels of the clusters for the different samples.

    This attribute is set during the learning phase.
    """

    n_clusters: int
    """The number of clusters.

    This attribute is set during the learning phase.
    """

    Settings: ClassVar[type[BaseClustererSettings]] = BaseClustererSettings

    def _post_init(self):
        super()._post_init()
        self.labels = array([])
        self.n_clusters = 0

    def _learn(
        self,
        indices: Sequence[int],
        fit_transformers: bool,
    ) -> None:
        super()._learn(indices, fit_transformers=fit_transformers)
        self.n_clusters = unique(self.labels).shape[0]
        if not self.n_clusters:
            msg = f"{self.__class__.__name__}._fit() did not set the labels attribute."
            raise NotImplementedError(msg)
