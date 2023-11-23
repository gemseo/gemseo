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
r"""Draw a scatter matrix from a :class:`.Dataset`.

The :class:`.ScatterMatrix` class implements the scatter plot matrix,
which is a way to visualize :math:`n` samples of a
multi-dimensional vector

.. math::

   x=(x_1,x_2,\ldots,x_d)\in\mathbb{R}^d

in several 2D subplots where the (i,j) subplot represents the cloud
of points

.. math::

   \left(x_i^{(k)},x_j^{(k)}\right)_{1\leq k \leq n}

while the (i,i) subplot represents the empirical distribution of the samples

.. math::

   x_i^{(1)},\ldots,x_i^{(n)}

by means of an histogram or a kernel density estimator.

A variable name can be passed to the :meth:`.DatasetPlot.execute` method
by means of the ``classifier`` keyword in order to color the curves
according to the value of the variable name. This is useful when the data is
labeled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Sequence

from gemseo.post.dataset.dataset_plot import DatasetPlot

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset


class ScatterMatrix(DatasetPlot):
    """Scatter plot matrix."""

    def __init__(
        self,
        dataset: Dataset,
        variable_names: Sequence[str] | None = None,
        classifier: str | None = None,
        kde: bool = False,
        size: int = 25,
        marker: str = "o",
        plot_lower: bool = True,
        plot_upper: bool = True,
    ) -> None:
        """
        Args:
            classifier: The name of the variable to build the cluster.
            kde: The type of the distribution representation.
                If ``True``, plot kernel-density estimator on the diagonal.
                Otherwise, use histograms.
            size: The size of the points.
            marker: The marker for the points.
            plot_lower: Whether to plot the lower part.
            plot_upper: Whether to plot the upper part.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            variable_names=variable_names,
            classifier=classifier,
            kde=kde,
            size=size,
            marker=marker,
            plot_lower=plot_lower,
            plot_upper=plot_upper,
        )

    def _create_specific_data_from_dataset(self) -> tuple[tuple[str, str, int] | None]:
        """
        Returns:
            The column of the dataset associated with the classifier
            if the classifier is not ``None``.

        Raises:
            ValueError: When the classifier does not exist.
        """  # noqa: D205, D212, D415
        classifier = self._specific_settings.classifier
        if classifier is not None and classifier not in self.dataset.variable_names:
            raise ValueError(
                f"{classifier} cannot be used as a classifier "
                f"because it is not a variable name; "
                f"available ones are: {self.dataset.variable_names}."
            )

        if self._specific_settings.classifier is not None:
            return (self._get_label(self._specific_settings.classifier)[1],)

        return (None,)
