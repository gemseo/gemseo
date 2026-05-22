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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Self-organizing map (SOM) to display high-dimensional design spaces."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from matplotlib import pyplot as plt
from minisom import MiniSom
from numpy import array
from numpy import float64
from numpy import full
from numpy import nan
from numpy import nanmax
from numpy import nanmin
from numpy import unique

from gemseo.post.base_post import BasePost
from gemseo.post.core.colormaps import PARULA
from gemseo.post.som_settings import SOM_Settings
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    import minisom
    from matplotlib.axes import Axes
    from numpy import ndarray

    from gemseo.typing import IntegerArray
    from gemseo.typing import RealArray


class SOM(BasePost[SOM_Settings]):
    """Self-organizing map (SOM).

    The goal is to plot every output over a two-dimensional design space.
    Two points that are close together in this space are also close together
    in the space combining the design space and the output space.
    """

    settings_class: ClassVar[type[SOM_Settings]] = SOM_Settings
    __CMAP: Final[tuple[str, tuple[tuple[float, float, float], ...]]] = PARULA

    @staticmethod
    def __train_som(data: RealArray, n_x: int = 5, n_y: int = 5) -> minisom.MiniSom:
        """Train the SOM.

        Args:
            data: The data to train the SOM, shaped as (n_samples, n_features).
            n_x: The number of neurons in the horizontal direction.
            n_y: The number of neurons in the vertical direction.

        Returns:
            The SOM.
        """
        som = MiniSom(n_x + 1, n_y + 1, data.shape[1])
        som.pca_weights_init(data)
        som.train(data, 1000)
        return som

    def _plot(self, settings: SOM_Settings) -> None:
        annotate = settings.annotate
        expected_output_names = [
            self._optimization_metadata.standardized_objective_name,
            *(
                name
                for name in self._optimization_metadata.output_name_to_constraint_names
            ),
        ]
        available_output_names = self.__get_available_output_names()
        # Exclude the output names without data in the dataset.
        for output_name in tuple(expected_output_names):
            if output_name not in available_output_names:
                expected_output_names.remove(output_name)

        x, y, cluster = self.__compute(settings.n_x, settings.n_y)

        output_dimension = self._dataset.get_view(
            variable_names=expected_output_names
        ).shape[1]
        n_cols = 3
        n_rows = output_dimension // n_cols
        if (output_dimension % n_cols) > 0:
            n_rows += 1

        fig, axes = plt.subplots(
            n_rows, n_cols, sharex=True, sharey=True, figsize=settings.fig_size
        )
        fig.suptitle("Self Organizing Maps of the design space", fontsize=14)

        index = 0
        axes = axes.ravel()
        for output_name in expected_output_names:
            output_data = self._dataset.get_view(variable_names=output_name).to_numpy()
            size = output_data.shape[1]
            for output_index, output_data_ in enumerate(output_data.T):
                self.__add_sub_plot(
                    x,
                    y,
                    cluster,
                    output_data_,
                    repr_variable(output_name, output_index, size=size),
                    index + 1,
                    axes[index],
                    annotate,
                )
                index += 1

        for ax in axes[index:]:
            ax.set_visible(False)

        fig.tight_layout()
        self._add_figure(fig)

    def __add_sub_plot(
        self,
        x: IntegerArray,
        y: IntegerArray,
        clusters: IntegerArray,
        output_data: RealArray,
        output_name: str,
        index: int,
        ax: Axes,
        annotate: bool,
    ) -> None:
        """Create the sub-plot associated with an output component.

        Args:
            x: The x samples.
            y: The y samples.
            clusters: The cluster samples.
            output_data: The samples of this output component
            output_name: The name of this output component.
            index: The index of the sub-plot.
            ax: The axes.
            annotate: Whether to display the average output value for every cell.
        """
        cluster_to_average = {}
        for cluster in unique(clusters):
            cluster_to_average[cluster] = output_data[clusters == cluster].mean()

        n_x = x.max()
        n_y = y.max()
        xy_to_average = full((n_x, n_y), nan, dtype=float64)
        for x_, y_, cluster in zip(x, y, clusters, strict=True):
            xy_to_average[int(x_) - 1, int(y_) - 1] = cluster_to_average[cluster]

        self.materials_for_plotting[index] = xy_to_average
        minimum_average_output = nanmin(xy_to_average)
        maximum_average_output = nanmax(xy_to_average)
        im1 = ax.imshow(
            xy_to_average,
            vmin=minimum_average_output - 0.01 * abs(minimum_average_output),
            vmax=maximum_average_output + 0.01 * abs(maximum_average_output),
            cmap=self.__CMAP,
            interpolation="nearest",
            aspect="auto",
        )

        font_size = 12
        if annotate:
            average_output_format = "%1.2g"
            for x, row in enumerate(xy_to_average):
                for y, cell in enumerate(row):
                    _ = ax.text(
                        y,
                        x,
                        average_output_format % cell,
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=font_size,
                    )

        ax.set_title(output_name, fontsize=font_size)
        plt.colorbar(im1)
        im1.axes.get_xaxis().set_visible(False)
        im1.axes.get_yaxis().set_visible(False)

    def __compute(self, n_x: int, n_y: int) -> tuple[ndarray, ndarray, ndarray]:
        """Build the SOM from the dataset.

        Args:
            n_x: The number of neurons in the horizontal direction.
            n_y: The number of neurons in the vertical direction.

        Returns:
            The x samples, the y samples and the cluster samples.
        """
        input_names = self._dataset.design_variable_names
        output_names = self.__get_available_output_names()
        normalize_samples = (
            self._dataset
            .get_view(variable_names=input_names + output_names)
            .get_normalized(use_min_max=False, center=True, scale=True)
            .to_numpy()
        )
        som = self.__train_som(normalize_samples, n_x, n_y)
        xy = array([som.winner(sample) for sample in normalize_samples])
        x = xy[:, 0]
        y = xy[:, 1]
        som_shape = som.get_weights().shape[:2]
        cluster_indices = array([x * som_shape[1] + y for x, y in xy])
        self._dataset.add_variable(
            variable_name="SOM_indx",
            data=cluster_indices,
            group_name=self._dataset.FUNCTION_GROUP,
        )
        self._dataset.add_variable(
            variable_name="SOM_i",
            data=x,
            group_name=self._dataset.FUNCTION_GROUP,
        )
        self._dataset.add_variable(
            variable_name="SOM_j",
            data=y,
            group_name=self._dataset.FUNCTION_GROUP,
        )
        return x, y, cluster_indices

    def __get_available_output_names(self) -> list[str]:
        """Return the available output names from the dataset.

        Returns:
            The available output names.
        """
        return (
            self._dataset.equality_constraint_names
            + self._dataset.inequality_constraint_names
            + self._dataset.objective_names
            + self._dataset.observable_names
        )
