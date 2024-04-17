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
"""The base class for clustering algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Union

from numpy import ndarray
from numpy import unique

from gemseo.mlearning.core.algos.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.algos.ml_algo import SavedObjectType as MLAlgoSavedObjectType
from gemseo.mlearning.core.algos.ml_algo import TransformerType
from gemseo.mlearning.core.algos.unsupervised import BaseMLUnsupervisedAlgo

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.datasets.dataset import Dataset

SavedObjectType = Union[MLAlgoSavedObjectType, ndarray, int]


class BaseClusterer(BaseMLUnsupervisedAlgo):
    """The base class for clustering algorithms."""

    labels: list[int]
    """The indices of the clusters for the different samples."""

    n_clusters: int
    """The number of clusters."""

    def __init__(  # noqa: D107
        self,
        data: Dataset,
        transformer: TransformerType = BaseMLUnsupervisedAlgo.IDENTITY,
        var_names: Iterable[str] | None = None,
        **parameters: MLAlgoParameterType,
    ) -> None:
        super().__init__(
            data, transformer=transformer, var_names=var_names, **parameters
        )
        self.labels = None
        self.n_clusters = None

    def _learn(
        self,
        indices: Sequence[int] | None,
        fit_transformers: bool,
    ) -> None:
        super()._learn(indices, fit_transformers=fit_transformers)
        if self.labels is None:
            msg = "self._fit() shall assign labels."
            raise ValueError(msg)
        self.n_clusters = unique(self.labels).shape[0]

    def _get_objects_to_save(self) -> dict[str, SavedObjectType]:
        objects = super()._get_objects_to_save()
        objects["labels"] = self.labels
        objects["n_clusters"] = self.n_clusters
        return objects
