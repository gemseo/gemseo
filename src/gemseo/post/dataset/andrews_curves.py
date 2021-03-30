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
Andrews curves
==============

The :class:`.AndrewsCurves` class implements the Andrew plot,
a.k.a. Andrews curves,
which is a way to visualize :math:`n` samples of a high-dimensional vector

.. math::

   x=(x_1,x_2,\ldots,x_d)\in\mathbb{R}^d

in a 2D referential by projecting each sample

.. math::

   x^{(i)}=(x_1^{(i)},x_2^{(i)},\ldots,x_d^{(i)})

onto the vector

.. math::

   \left(\frac{1}{\sqrt{2}},\sin(t),\cos(t),\sin(2t),\cos(2t), \ldots\right)

which is composed of the :math:`d` first elements of the Fourier series:

.. math::

   f_i(t)=\left(\frac{x_1^{(i)}}{\sqrt{2}},x_2^{(i)}\sin(t),x_3^{(i)}\cos(t),
   x_4^{(i)}\sin(2t),x_5^{(i)}\cos(2t),\ldots\right)

Each curve :math:`t\mapsto f_i(t)` is plotted
over the interval :math:`[-\pi,\pi]`
and structure in the data may be visible in these :math:`n` Andrews curves.

A variable name can be passed to the :meth:`.DatasetPlot.execute` method
by means of the :code:`classifier` keyword in order to color the curves
according to the value of the variable name. This is useful when the data is
labeled.
"""
from __future__ import absolute_import, division, unicode_literals

import matplotlib.pyplot as plt
from future import standard_library
from pandas.plotting import andrews_curves

from gemseo.post.dataset.dataset_plot import DatasetPlot

standard_library.install_aliases()


class AndrewsCurves(DatasetPlot):
    """ Andrews curves. """

    def _plot(self, classifier):
        """Andrews curves.

        :param classifier: variable name to build the cluster.
        :type classifier: str
        """
        if classifier not in self.dataset.variables:
            raise ValueError(
                "Classifier must be one of these names: "
                + ", ".join(self.dataset.variables)
            )

        dataframe = self.dataset.export_to_dataframe()
        label, varname = self._get_label(classifier)
        if self.dataset.strings_encoding[label]:
            for comp, codes in self.dataset.strings_encoding[label].items():
                column = (self.dataset.get_group(label), label, str(comp))
                for key, value in codes.items():
                    dataframe.loc[dataframe[column] == key, column] = value
        andrews_curves(dataframe, varname)
        fig = plt.gcf()
        return fig
