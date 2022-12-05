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
r"""Draw a radar visualization from a :class:`.Dataset`.

The :class:`.Radar` class implements the Radviz plot,
which is a way to visualize :math:`n` samples of a multi-dimensional vector

.. math::

   x=(x_1,x_2,\ldots,x_d)\in\mathbb{R}^d

in a 2D referential and to highlight the separability of the data.

For that, each sample

.. math::

   x^{(i)}=(x_1^{(i)},x_2^{(i)},\ldots,x_d^{(i)})

is rendered inside the unit disc
with the influences of the different parameters evenly distributed
on its circumference. Each parameter influence varies from 0 to 1
and can be interpreted compared to the others.

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
from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.plotting import radviz

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot


class Radar(DatasetPlot):
    """Radar visualization."""

    def __init__(
        self,
        dataset: Dataset,
        classifier: str,
    ) -> None:
        """
        Args:
            classifier: The name of the variable to group the data.
        """  # noqa: D205, D212, D415
        super().__init__(dataset, classifier=classifier)

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        classifier = self._param.classifier
        if classifier not in self.dataset.variables:
            raise ValueError(
                "Classifier must be one of these names: "
                + ", ".join(self.dataset.variables)
            )

        dataframe = self.dataset.export_to_dataframe()
        label, _ = self._get_label(classifier)
        if self.dataset.strings_encoding[label]:
            for comp, codes in self.dataset.strings_encoding[label].items():
                column = (self.dataset.get_group(label), label, str(comp))
                for key, value in codes.items():
                    dataframe.loc[dataframe[column] == key, column] = value

        dataframe.columns = self._get_variables_names(dataframe)
        fig, axes = self._get_figure_and_axes(fig, axes)
        radviz(dataframe, label, ax=axes)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title)
        return [fig]
