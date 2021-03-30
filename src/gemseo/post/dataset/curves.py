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
"""
Curve plot
==========

A :class:`.Curves` plot represents samples of a functional variable
:math:`y(x)` discretized over a 1D mesh. Both evaluations of :math:`y`
and mesh are stored in a :class:`.Dataset`, :math:`y` as a parameter
and the mesh as a metadata.
"""
from __future__ import absolute_import, division, unicode_literals

import matplotlib.pyplot as plt
from future import standard_library

from gemseo.post.dataset.dataset_plot import DatasetPlot

standard_library.install_aliases()


class Curves(DatasetPlot):
    """ Plot curves y_i over the mesh x. """

    def _plot(self, mesh, variable, samples=None, color="blue"):
        """Curve.

        :param mesh: name of the mesh stored in Dataset.metadata.
        :type mesh: str
        :param variable: variable name for the x-axis.
        :type variable: float
        :param samples: samples indices. If None, plot all samples.
            Default: None.
        :type samples: list(int)
        :param color: line color. Default: 'blue'.
        :type color: str
        """

        def lines_gen():
            """ Linestyle generator. """
            yield "-"
            i = 1
            for i in range(1, self.dataset.n_samples):
                yield (0, (i, 1, 1, 1))

        if samples is not None:
            output = self.dataset[variable][variable][samples, :].T
        else:
            output = self.dataset[variable][variable].T
        n_samples = output.shape[1]

        lines = lines_gen()
        for sample in range(n_samples):
            plt.plot(
                self.dataset.metadata[mesh],
                output[:, sample],
                linestyle=next(lines),
                color=color,
                label=sample,
            )
        plt.xlabel(mesh)
        plt.ylabel(variable + "(" + mesh + ")")
        plt.legend()
        fig = plt.gcf()
        return fig
