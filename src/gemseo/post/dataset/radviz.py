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
by means of the ``classifier`` keyword in order to color the curves
according to the value of the variable name. This is useful when the data is
labeled or when we are looking for the samples for which the classifier value
is comprised in some interval specified by the ``lower`` and ``upper``
arguments
(default values are set to ``-inf`` and ``inf`` respectively).
In the latter case, the color scale is composed of only two values: one for
the samples positively classified and one for the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset


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

    def _create_specific_data_from_dataset(self) -> tuple[Dataset, str]:
        """
        Returns:
            The dataset with decoded values,
            the name of the classifier.
        """  # noqa: D205, D212, D415
        classifier = self._specific_settings.classifier
        if classifier not in self.dataset.variable_names:
            raise ValueError(
                f"The classifier ({classifier}) is not stored in the dataset; "
                "available variables are "
                f"{pretty_str(self.dataset.variable_names, use_and=True)}."
            )

        dataset = self.dataset
        label = self._get_label(classifier)[0]
        str_encoder = self.dataset.misc.get("labels", {})
        if len(str_encoder):
            for variable, meaning in str_encoder.items():
                data = self.dataset.get_view(variable_names=variable).to_numpy()
                self.dataset.update_data(meaning[data.ravel()], variable_names=variable)

        dataset.columns = self._get_variable_names(dataset)
        return dataset.reindex(sorted(dataset.columns), axis=1), label
