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
"""The base class to assess the quality of a clusterer."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.mlearning.core.quality.base_ml_algo_quality import BaseMLAlgoQuality
from gemseo.mlearning.core.quality.base_ml_algo_quality import MeasureType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.mlearning.clustering.algos.base_clusterer import BaseClusterer


class BaseClustererQuality(BaseMLAlgoQuality):
    """The base class to assess the quality of a clusterer."""

    algo: BaseClusterer

    def __init__(
        self,
        algo: BaseClusterer,
        fit_transformers: bool = BaseMLAlgoQuality._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for clustering.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers=fit_transformers)

    def compute_learning_measure(  # noqa: D102
        self,
        samples: Sequence[int] = (),
        multioutput: bool = True,
    ) -> MeasureType:
        return self._compute_measure(
            self._get_data()[self._pre_process(samples)[0]],
            self.algo.labels,
            multioutput,
        )

    @abstractmethod
    def _compute_measure(
        self,
        data: ndarray,
        labels: ndarray,
        multioutput: bool = True,
    ) -> MeasureType:
        """Compute the quality measure.

        Args:
            data: The reference data.
            labels: The predicted labels.
            multioutput: Whether the quality measure is returned
                for each component of the outputs.
                Otherwise, the average quality measure.

        Returns:
            The quality measure.
        """

    def _get_data(self) -> ndarray:
        """Get data.

        Returns:
            The learning data.
        """
        return self.algo.learning_set.get_view(
            variable_names=self.algo.var_names
        ).to_numpy()
