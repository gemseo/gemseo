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
Parallel coordinates plot
=========================

The :class:`.ParallelCoordinates` class implements the parallel coordinates
plot, a.k.a. cowebplot, which is a way to visualize :math:`n` samples of a
high-dimensional vector

.. math::

   x=(x_1,x_2,\ldots,x_d)\in\mathbb{R}^d

in a 2D referential by representing each sample

.. math::

   x^{(i)}=(x_1^{(i)},x_2^{(i)},\ldots,x_d^{(i)})

as a piece-wise line where the x-values of the nodes from left to right
are the values of :math:`x_1`, :math:`x_2`, ... and :math:`x_d^{(i)}`.

A variable name is required by the :meth:`.DatasetPlot.execute` method
by means of the :code:`classifier` keyword in order to color the curves
according to the value of the variable name. This is useful when the data is
labeled or when we are looking for the samples for which the classifier value
is comprised in some interval specified by the :code:`lower` and :code:`upper`
arguments
(default values are set to :code:`-inf` and :code:`inf` respectively).
In the latter case, the color scale is composed of only two values: one for
the samples positively classified and one for the others.
"""
from __future__ import absolute_import, division, unicode_literals

import matplotlib.pyplot as plt
from future import standard_library
from numpy import inf
from pandas.plotting import parallel_coordinates

from gemseo.post.dataset.dataset_plot import DatasetPlot

standard_library.install_aliases()


class ParallelCoordinates(DatasetPlot):
    """ Parallel coordinates. """

    def _plot(self, classifier, lower=-inf, upper=inf, **kwargs):
        """Parallel coordinates.

        :param classifier: variable name to build the cluster.
        :type classifier: str
        :param lower: lower bound of the cluster. Default: -infinity.
        :type lower: float
        :param upper: upper bound of the cluster. Default: +infinity.
        :type upper: float
        """
        if classifier not in self.dataset.variables:
            raise ValueError(
                "Classifier must be one of these names: "
                + ", ".join(self.dataset.variables)
            )
        label, varname = self._get_label(classifier)
        dataframe = self.dataset.export_to_dataframe()
        cluster = varname
        columns = list(dataframe.columns)

        def is_btw(row):
            return lower < row[varname] < upper

        if lower != -inf or upper != inf:
            cluster = str(lower) + " < " + label + " < " + str(upper)
            cluster = ("classifiers", cluster, "0")
            dataframe[cluster] = dataframe.apply(is_btw, axis=1)
        p_c = parallel_coordinates(dataframe, cluster, cols=columns, **kwargs)
        p_c.set_xticklabels(self._get_varnames(columns))
        if lower != -inf or upper != inf:
            plt.title("Cobwebplot based on the classifier: " + cluster[1])
        else:
            plt.gca().get_legend().remove()
        fig = plt.gcf()
        return fig
