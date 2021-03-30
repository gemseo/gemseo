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
r"""
Scatter matrix
==============

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
from __future__ import absolute_import, division, unicode_literals

import matplotlib.pyplot as plt
from future import standard_library

try:
    from pandas.plotting import scatter_matrix
except ImportError:
    from pandas import scatter_matrix

from gemseo.post.dataset.dataset_plot import DatasetPlot

standard_library.install_aliases()


class ScatterMatrix(DatasetPlot):
    """ Scatter plot matrix. """

    def _plot(
        self,
        classifier=None,
        kde=False,
        size=25,
        marker="o",
        figsize_x=10,
        figsize_y=10,
    ):
        """Scatter matrix plot.

        :param classifier: variable name to build the cluster. Default: None
        :type classifier: str
        :param kde: if True, plot kernel-density estimator on the diagonal.
            Otherwise, use histograms. Default: False.
        :type kde: bool
        :param figsize_x: size of figure in horizontal direction (inches)
        :type figsize_x: int
        :param figsize_y: size of figure in vertical direction (inches)
        :type figsize_y: int
        """
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
        fig = plt.gcf()
        return fig

    def _scatter_matrix_for_group(
        self, classifier, dataframe, diagonal, size, marker, figsize_x, figsize_y
    ):
        """Scatter matrix plot for group.

        :param classifier: variable name to build the cluster. If None,
            use the first group name if the dataset is a GroupDataset,
            the first output name if the dataset is an IODataset,
            otherwise the last parameter name.
        :type classifier: str
        :param dataframe: pandas dataframe
        :param str diagonal: if 'kde', plot kernel-density estimator
            on the diagonal. If 'hist', use histograms.
        :param int figsize_x: size of figure in horizontal direction (inches)
        :param int figsize_y: size of figure in vertical direction (inches)
        """
        palette = dict(enumerate("bgrcmyk"))
        groups = self.dataset.get_data_by_names([classifier], False)[:, 0:1]
        colors = [palette[group[0] % len(palette)] for group in groups]
        _, varname = self._get_label(classifier)
        dataframe = dataframe.drop(varname, 1)
        dataframe.columns = self._get_varnames(dataframe)
        scatter_matrix(
            dataframe,
            diagonal=diagonal,
            color=colors,
            s=size,
            marker=marker,
            figsize=(figsize_x, figsize_y),
        )

    def _scatter_matrix(self, dataframe, diagonal, size, marker, figsize_x, figsize_y):
        """Scatter matrix plot for group.

        :param dataframe: pandas dataframe
        :param str diagonal: if 'kde', plot kernel-density estimator
            on the diagonal. If 'hist', use histograms.
        :param int figsize_x: size of figure in horizontal direction (inches)
        :param int figsize_y: size of figure in vertical direction (inches)
        """
        dataframe.columns = self._get_varnames(dataframe)
        scatter_matrix(
            dataframe,
            diagonal=diagonal,
            figsize=(figsize_x, figsize_y),
            s=size,
            marker=marker,
        )
