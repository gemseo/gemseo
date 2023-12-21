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
"""Draw curves from a :class:`.Dataset`.

A :class:`.Curves` plot represents samples of a functional variable
:math:`y(x)` discretized over a 1D mesh. Both evaluations of :math:`y`
and mesh are stored in a :class:`.Dataset`, :math:`y` as a parameter
and the mesh as a misc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.dataset_plot import DatasetPlot

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from gemseo.datasets.dataset import Dataset


class Curves(DatasetPlot):
    """Plot curves y_i over the mesh x."""

    def __init__(
        self,
        dataset: Dataset,
        mesh: str,
        variable: str,
        samples: Sequence[int] | None = None,
    ) -> None:
        """
        Args:
            mesh: The name of the dataset misc corresponding to the mesh.
            variable: The name of the variable for the x-axis.
            samples: The indices of the samples to plot.
                If ``None``, plot all the samples.
        """  # noqa: D205, D212, D415
        super().__init__(dataset, mesh=mesh, variable=variable, samples=samples)

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[NDArray[float], list[str]]:
        """
        Returns:
            The values of the points of the curves on the y-axis (one curve per row),
            the labels of the curves.
        """  # noqa: D205 D212 D415
        samples = self._specific_settings.samples
        y_values = self.dataset.get_view(
            variable_names=self._specific_settings.variable
        ).to_numpy()
        if samples is None:
            self._n_items = len(y_values)
            samples = range(self._n_items)
        else:
            self._n_items = len(samples)
            y_values = y_values[samples, :]

        self._set_color(self._n_items)
        self._set_linestyle(self._n_items, list(self.__get_line_style_generator()))
        return y_values, [self.dataset.index[sample] for sample in samples]

    def __get_line_style_generator(
        self,
    ) -> Generator[str | tuple[int, tuple[int, int, int, int]], None, None]:
        """A line style generator.

        Yields:
            A line style.
        """
        yield "-"
        for i in range(1, self._n_items):
            yield 0, (i, 1, 1, 1)
