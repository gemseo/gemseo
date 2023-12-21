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
r"""Draw parallel coordinates from a :class:`.Dataset`.

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
from typing import Any

from numpy import inf

from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset


class ParallelCoordinates(DatasetPlot):
    """Parallel coordinates."""

    def __init__(
        self,
        dataset: Dataset,
        classifier: str,
        lower: float = -inf,
        upper: float = inf,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            classifier: The name of the variable to group the data.
            lower: The lower bound of the cluster.
            upper: The upper bound of the cluster.
            **kwargs: The options to pass to pandas.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset, classifier=classifier, lower=lower, upper=upper, kwargs=kwargs
        )

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[
        Dataset,
        tuple[str, str, int],
        list[tuple[str, str, int]],
        float,
        float,
        dict[str, Any],
    ]:
        """
        Returns:
            The dataset to be used,
            the identifier of the cluster.
        """  # noqa: D205, D212, D415
        classifier = self._specific_settings.classifier
        upper = self._specific_settings.upper
        lower = self._specific_settings.lower
        if classifier not in self.dataset.variable_names:
            raise ValueError(
                "Classifier must be one of these names: "
                f"{pretty_str(self.dataset.variable_names, use_and=True)}."
            )
        label, varname = self._get_label(classifier)
        dataframe = self.dataset.copy()
        cluster = varname

        def is_btw(row):
            return lower < row.loc[varname] < upper

        if lower != -inf or upper != inf:
            cluster = ("classifiers", f"{lower} < {label} < {upper}", "0")
            dataframe[cluster] = dataframe.apply(is_btw, axis=1)

        if lower != -inf or upper != inf:
            default_title = f"Cobweb plot based on the classifier: {cluster[1]}"
        else:
            default_title = None
        self.title = self.title or default_title
        return dataframe, cluster
