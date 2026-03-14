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
r"""Draw Andrews curves from a [Dataset][gemseo.datasets.dataset.Dataset].

The [AndrewsCurves][gemseo.post.dataset.andrews_curves.AndrewsCurves] class
implements the Andrew plot, a.k.a. Andrews curves,
which is a way to visualize $n$ samples of a high-dimensional vector

$$x=(x_1,x_2,\ldots,x_d)\in\mathbb{R}^d$$

in a 2D referential by projecting each sample

$$x^{(i)}=(x_1^{(i)},x_2^{(i)},\ldots,x_d^{(i)})$$

onto the vector

$$\left(\frac{1}{\sqrt{2}},\sin(t),\cos(t),\sin(2t),\cos(2t), \ldots\right)$$

which is composed of the $d$ first elements of the Fourier series:

$$
   f_i(t)=\left(\frac{x_1^{(i)}}{\sqrt{2}},x_2^{(i)}\sin(t),x_3^{(i)}\cos(t),
   x_4^{(i)}\sin(2t),x_5^{(i)}\cos(2t),\ldots\right)
$$

Each curve $t\mapsto f_i(t)$ is plotted
over the interval $[-\pi,\pi]$
and structure in the data may be visible in these $n$ Andrews curves.

A variable name can be passed to the
[DatasetPlot.execute()][gemseo.post.dataset.base.BaseDatasetPlot.execute]
method by means of the `classifier` keyword
in order to color the curves according to the value of the variable name.
This is useful when the data is labeled.
"""

from __future__ import annotations

from gemseo.post.dataset.andrews_curves_settings import AndrewsCurves_Settings
from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.utils.string_tools import pretty_str


class AndrewsCurves(BaseDatasetPlot[AndrewsCurves_Settings]):
    """Andrews curves."""

    settings_class = AndrewsCurves_Settings

    def _create_specific_data_from_dataset(self) -> tuple[tuple[str, str, int]]:
        """
        Returns:
            The column of the dataset containing the group names.
        """  # noqa: D205 D212 D415
        classifier = self.settings.classifier
        if classifier not in self.dataset.variable_names:
            msg = (
                "Classifier must be one of these names: "
                f"{pretty_str(self.dataset.variable_names, use_and=True)}."
            )
            raise ValueError(msg)

        return (self._get_label(classifier)[1],)
