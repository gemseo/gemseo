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
r"""Draw a pair plot from a [Dataset][gemseo.datasets.dataset.Dataset].

The [PairPlot][gemseo.post.dataset.pair_plot.PairPlot] class
implements the pair plot, a.k.a. scatter plot matrix,
which is a way to visualize $n$ samples of a
multi-dimensional vector

$$x=(x_1,x_2,\ldots,x_d)\in\mathbb{R}^d$$

in several 2D subplots where the (i,j) subplot represents the cloud
of points

$$\left(x_i^{(k)},x_j^{(k)}\right)_{1\leq k \leq n}$$

while the (i,i) subplot represents the empirical distribution of the samples

$$x_i^{(1)},\ldots,x_i^{(n)}$$

by means of a histogram or a kernel density estimator.

A variable name is required by the
[DatasetPlot.execute()][gemseo.post.dataset.base.BaseDatasetPlot.execute] method
by means of the `classifier` keyword in order to color the curves
according to the value of the variable name. This is useful when the data is
labeled.
"""

from __future__ import annotations

from typing import ClassVar

from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.pair_plot_settings import PairPlot_Settings


class PairPlot(BaseDatasetPlot[PairPlot_Settings]):
    """Pair plot, a.k.a. pair plot."""

    settings_class: ClassVar[type[PairPlot_Settings]] = PairPlot_Settings

    def _create_specific_data_from_dataset(self) -> tuple[tuple[str, str, int] | None]:
        """
        Returns:
            The column of the dataset associated with the classifier
            if the classifier exists.

        Raises:
            ValueError: When the classifier does not exist.
        """  # noqa: D205, D212, D415
        classifier = self.settings.classifier
        if classifier and classifier not in self.dataset.variable_names:
            msg = (
                f"{classifier} cannot be used as a classifier "
                f"because it is not a variable name; "
                f"available ones are: {sorted(self.dataset.variable_names)}."
            )
            raise ValueError(msg)

        if classifier:
            return (self._get_label(classifier)[1],)

        return (None,)
