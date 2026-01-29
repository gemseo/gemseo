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

from gemseo.machine_learning.core.quality.base_ml_model_quality import (
    BaseMLModelQuality,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.machine_learning.clustering.models.base_clusterer import BaseClusterer
    from gemseo.machine_learning.core.quality.base_ml_model_quality import MeasureType


class BaseClustererQuality(BaseMLModelQuality):
    """The base class to assess the quality of a clusterer."""

    model: BaseClusterer

    def __init__(
        self,
        model: BaseClusterer,
        fit_transformers: bool = BaseMLModelQuality._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            model: A machine learning model for clustering.
        """  # noqa: D205 D212
        super().__init__(model, fit_transformers=fit_transformers)

    def compute_learning_measure(  # noqa: D102
        self,
        samples: Sequence[int] = (),
        multioutput: bool = True,
    ) -> MeasureType:
        return self._compute_measure(
            self._get_data()[self._pre_process(samples)[0]],
            self.model.labels,
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
        return self.model.learning_set.get_view(
            variable_names=self.model.var_names
        ).to_numpy()
