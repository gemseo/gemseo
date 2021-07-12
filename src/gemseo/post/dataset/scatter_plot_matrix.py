# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

from typing import List, Mapping, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame

from gemseo.post.dataset.dataset_plot import DatasetPlot

try:
    from pandas.plotting import scatter_matrix
except ImportError:
    from pandas import scatter_matrix


class ScatterMatrix(DatasetPlot):
    """Scatter plot matrix."""

    def _plot(
        self,
        properties,  # type: Mapping
        classifier=None,  # type: Optional[str]
        kde=False,  # type: bool
        size=25,  # type: int
        marker="o",  # type: str
    ):  # type: (...) -> List[Figure]
        """
        Args:
            classifier: The name of the variable to build the cluster.
            kde: The type of the distribution representation.
                If True, plot kernel-density estimator on the diagonal.
                Otherwise, use histograms.
            size: The size of the points.
            marker: The marker for the points.
        """
        figsize_x = properties.get(self.FIGSIZE_X) or 10
        figsize_y = properties.get(self.FIGSIZE_Y) or 10
        if classifier is not None and classifier not in self.dataset.variables:
            raise ValueError(
                "Classifier must be one of these names: "
                + ", ".join(self.dataset.variables)
            )
        if kde:
            diagonal = "kde"
        else:
            diagonal = "hist"
        dataframe = self.dataset.export_to_dataframe()
        if classifier is None:
            self._scatter_matrix(
                dataframe, diagonal, size, marker, figsize_x, figsize_y
            )
        else:
            self._scatter_matrix_for_group(
                classifier, dataframe, diagonal, size, marker, figsize_x, figsize_y
            )
        return [plt.gcf()]

    def _scatter_matrix_for_group(
        self,
        classifier,  # type: str
        dataframe,  # type: DataFrame
        diagonal,  # type: str
        size,  # type: int
        marker,  # type: str
        figsize_x,  # type: int
        figsize_y,  # type: int
    ):  # type: (...) -> None
        """Scatter matrix plot for group.

        Args:
            classifier: The name of the variable to group the data.
            dataframe: The data to plot.
            diagonal: The type of distribution representation, either "kde" or "hist".
            size: The size of the points.
            marker: The marker for the points.
            figsize_x: The size of the figure in horizontal direction (inches).
            figsize_y: The size of the figure in vertical direction (inches).
        """
        palette = dict(enumerate("bgrcmyk"))
        groups = self.dataset.get_data_by_names([classifier], False)[:, 0:1]
        colors = [palette[group[0] % len(palette)] for group in groups]
        _, varname = self._get_label(classifier)
        dataframe = dataframe.drop(varname, 1)
        dataframe.columns = self._get_variables_names(dataframe)
        scatter_matrix(
            dataframe,
            diagonal=diagonal,
            color=colors,
            s=size,
            marker=marker,
            figsize=(figsize_x, figsize_y),
        )

    def _scatter_matrix(
        self,
        dataframe,  # type: DataFrame
        diagonal,  # type: str
        size,  # type: int
        marker,  # type: str
        figsize_x,  # type: int
        figsize_y,  # type: int
    ):  # type: (...) -> None
        """Scatter matrix plot for group.

        Args:
            dataframe: The data to plot.
            diagonal: The type of distribution representation, either "kde" or "hist".
            size: The size of the points.
            marker: The marker for the points.
            figsize_x: The size of the figure in horizontal direction (inches).
            figsize_y: The size of the figure in vertical direction (inches).

        Returns:
            The figure.
        """
        dataframe.columns = self._get_variables_names(dataframe)
        scatter_matrix(
            dataframe,
            diagonal=diagonal,
            figsize=(figsize_x, figsize_y),
            s=size,
            marker=marker,
        )
