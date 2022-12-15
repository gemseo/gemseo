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
by means of the :code:`classifier` keyword in order to color the curves
according to the value of the variable name. This is useful when the data is
labeled.
"""
from __future__ import annotations

from typing import Sequence

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.plotting import scatter_matrix

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot


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
                If True, plot kernel-density estimator on the diagonal.
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

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        variable_names = self._param.variable_names
        classifier = self._param.classifier
        kde = self._param.kde
        size = self._param.size
        marker = self._param.marker
        if variable_names is None:
            variable_names = self.dataset.variables

        if classifier is not None and classifier not in self.dataset.variables:
            raise ValueError(
                f"{classifier} cannot be used as a classifier "
                f"because it is not a variable name; "
                f"available ones are: {self.dataset.variables}."
            )

        if kde:
            diagonal = "kde"
        else:
            diagonal = "hist"

        dataframe = self.dataset.export_to_dataframe(variable_names=variable_names)
        kwargs = {}
        if classifier is not None:
            palette = dict(enumerate("bgrcmyk"))
            groups = self.dataset.get_data_by_names([classifier], False)[:, 0:1]
            kwargs["color"] = [palette[group[0] % len(palette)] for group in groups]
            _, variable_name = self._get_label(classifier)
            dataframe = dataframe.drop(labels=variable_name, axis=1)

        dataframe.columns = self._get_variables_names(dataframe)
        fig, axes = self._get_figure_and_axes(fig, axes, self.fig_size)
        sub_axes = scatter_matrix(
            dataframe,
            diagonal=diagonal,
            s=size,
            marker=marker,
            figsize=self.fig_size,
            ax=axes,
            **kwargs,
        )

        n_cols = sub_axes.shape[0]
        if not (self._param.plot_lower and self._param.plot_upper):
            for i in range(n_cols):
                for j in range(n_cols):
                    sub_axes[i, j].get_xaxis().set_visible(False)
                    sub_axes[i, j].get_yaxis().set_visible(False)

        if not self._param.plot_lower:
            for i in range(n_cols):
                for j in range(i):
                    sub_axes[i, j].set_visible(False)

            for i in range(n_cols):
                sub_axes[i, i].get_xaxis().set_visible(True)
                sub_axes[i, i].get_yaxis().set_visible(True)

        if not self._param.plot_upper:
            for i in range(n_cols):
                for j in range(i + 1, n_cols):
                    sub_axes[i, j].set_visible(False)

            for i in range(n_cols):
                sub_axes[-1, i].get_xaxis().set_visible(True)
                sub_axes[i, 0].get_yaxis().set_visible(True)

        plt.suptitle(self.title)
        return [fig]
