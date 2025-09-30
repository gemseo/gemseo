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
"""Self Organizing Maps to display high dimensional design spaces."""

from __future__ import annotations

import logging
from math import floor
from math import sqrt
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

import numpy as np
from matplotlib import pyplot as plt
from minisom import MiniSom
from numpy import array
from numpy import bincount
from numpy import float64
from numpy import int32
from numpy import isnan
from numpy import logical_not
from numpy import max as np_max
from numpy import mean
from numpy import mgrid
from numpy import min as np_min
from numpy import ndarray
from numpy import nonzero
from numpy import unique
from numpy import zeros

from gemseo.post.base_post import BasePost
from gemseo.post.core.colormaps import PARULA
from gemseo.post.som_settings import SOM_Settings

if TYPE_CHECKING:
    import minisom
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from gemseo.typing import IntegerArray
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)


class SOM(BasePost[SOM_Settings]):
    """Self organizing map clustering optimization history.

    Options of the plot method are the x- and y- numbers of cells in the SOM.
    """

    Settings: ClassVar[type[SOM_Settings]] = SOM_Settings
    __CMAP: Final[tuple[str, tuple[tuple[float, float, float], ...]]] = PARULA

    @staticmethod
    def __build_som_from_vars(
        data: RealArray,
        som_grid_nx: int = 5,
        som_grid_ny: int = 5,
        n_iterations: int = 1000,
        verbose: bool = False,
    ) -> minisom.MiniSom:
        """Builds the SOM from the design variables history.

        Args:
            data: The data history (n_iter,n_dv).
            som_grid_nx: The number of neurons in the x direction.
            som_grid_ny: The number of neurons in the y direction.
            verbose: The verbose for SOM training.

        Returns:
            The self organizing map
        """
        LOGGER.info("Building Self Organizing Map from optimization history:")
        LOGGER.info("    Number of neurons in x direction = %s", som_grid_nx)
        LOGGER.info("    Number of neurons in y direction = %s", som_grid_ny)

        var_som = MiniSom(som_grid_nx + 1, som_grid_ny + 1, data.shape[1])
        var_som.pca_weights_init(data)
        var_som.train(data, n_iterations, verbose=verbose)

        return var_som

    def _plot(self, settings: SOM_Settings) -> None:
        n_x = settings.n_x
        n_y = settings.n_y
        annotate = settings.annotate

        criteria = [
            self._optimization_metadata.standardized_objective_name,
            *(
                name
                for name in self._optimization_metadata.output_names_to_constraint_names
            ),
        ]
        # all_data = self.database.get_function_names()
        all_data = (
            self._dataset.equality_constraint_names
            + self._dataset.inequality_constraint_names
            + self._dataset.objective_names
            + self._dataset.observable_names
        )
        # Ensure that the data is available in the dataset
        for criterion in tuple(criteria):
            if criterion not in all_data:
                criteria.remove(criterion)
        figure = plt.figure(figsize=settings.fig_size)
        figure.suptitle("Self Organizing Maps of the design space", fontsize=14)
        subplot_number = 0
        self.__compute(n_x, n_y)
        for criterion in criteria:
            f_hist = list(
                self._dataset.get_view(
                    variable_names=[
                        "SOM_i",
                        "SOM_j",
                        "SOM_indx",
                        criterion,
                    ]
                ).to_numpy(dtype=object)
            )

            if len(f_hist[0]) > 4:
                for i in range(len(f_hist)):
                    f_hist[i] = np.array(
                        [f_hist[i][0], f_hist[i][1], f_hist[i][2], f_hist[i][3:]],
                        dtype=object,
                    )

            if isinstance(f_hist[0][3], ndarray):
                dim_val = f_hist[0][3].size
                for _ in range(dim_val):
                    subplot_number += 1

            else:
                subplot_number += 1

        grid_size_x = 3
        grid_size_y = subplot_number // grid_size_x
        if (subplot_number % grid_size_x) > 0:
            grid_size_y += 1

        index = 0
        for criterion in criteria:
            f_hist = list(
                self._dataset.get_view(
                    variable_names=[
                        "SOM_i",
                        "SOM_j",
                        "SOM_indx",
                        criterion,
                    ]
                ).to_numpy(dtype=object)
            )
            if len(f_hist[0]) > 4:
                for i in range(len(f_hist)):
                    f_hist[i] = np.array(
                        [f_hist[i][0], f_hist[i][1], f_hist[i][2], f_hist[i][3:]],
                        dtype=object,
                    )

            if isinstance(f_hist[0][3], ndarray):
                for k in range(f_hist[0][3].size):
                    self.__plot_som_from_scalar_data(
                        [[*f_h[0:3], f_h[3][k]] for f_h in f_hist],
                        f"{criterion}[{k}]",
                        index + 1,
                        grid_size_x=grid_size_x,
                        grid_size_y=grid_size_y,
                        annotate=annotate,
                    )
                    index += 1

            else:
                self.__plot_som_from_scalar_data(
                    f_hist,
                    criterion,
                    index + 1,
                    grid_size_x=grid_size_x,
                    grid_size_y=grid_size_y,
                    annotate=annotate,
                )
                index += 1

        figure.tight_layout()
        self._add_figure(figure)

    def __plot_som_from_scalar_data(
        self,
        f_hist_scalar: ArrayLike,
        criteria: str,
        fig_indx: int,
        grid_size_x: int = 3,
        grid_size_y: int = 20,
        annotate: bool = False,
    ) -> Axes:
        """Builds the SOM plot after computation for a given criteria.

        Args:
            criteria: The criteria to show.
            f_hist_scalar: The scalar data to show.
            fig_indx: The axe index in the figure.
            grid_size_x: The number of SOMs in the grid on the x axis.
            grid_size_y: The number of SOMs in the grid on the y axis.
            annotate: If ``True``, add label with average value of neural.
        """
        f_hist = array(f_hist_scalar).T.real
        unique_ind = unique(f_hist[2, :])
        average = {}
        for som_id in unique_ind:
            where_somid = (f_hist[2, :] == som_id).nonzero()[0]
            ranges_of_uniques = f_hist[3, where_somid]
            average[som_id] = mean(ranges_of_uniques)

        ijshape = array((np_max(f_hist[0, :]), np_max(f_hist[1, :])), dtype=int32)
        mat_ij = zeros(ijshape, dtype=float64)
        mat_ij[:, :] = float("nan")
        for itr in range(f_hist.shape[-1]):
            i, j, somindx, _ = f_hist[:, itr]
            mat_ij[int(i) - 1, int(j) - 1] = average[somindx]
        empty = isnan(mat_ij)
        non_empty = logical_not(empty)
        ax = plt.subplot(grid_size_y, grid_size_x, fig_indx)
        minv = np_min(mat_ij[non_empty])
        maxv = np_max(mat_ij[non_empty])
        self.materials_for_plotting[fig_indx] = mat_ij
        im1 = ax.imshow(
            mat_ij,
            vmin=minv - 0.01 * abs(minv),
            vmax=maxv + 0.01 * abs(maxv),
            cmap=self.__CMAP,
            interpolation="nearest",
            aspect="auto",
        )  # "spectral" "hot" "RdBu_r"

        if annotate:
            crit_format = "%1.2g"
            for i in range(mat_ij.shape[0]):
                for j in range(mat_ij.shape[1]):
                    _ = ax.text(
                        j,
                        i,
                        crit_format % mat_ij[i, j],
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=12,
                    )

        ax.set_title(criteria, fontsize=12)
        plt.colorbar(im1)
        im1.axes.get_xaxis().set_visible(False)
        im1.axes.get_yaxis().set_visible(False)
        return ax

    def __compute(
        self,
        som_grid_nx: int = 5,
        som_grid_ny: int = 5,
    ) -> None:
        """Build the SOM from optimization history.

        Args:
            som_grid_nx: The number of neurons in the x direction.
            som_grid_ny: The number of neurons in the y direction.
        """
        function_names = (
            self._dataset.equality_constraint_names
            + self._dataset.inequality_constraint_names
            + self._dataset.objective_names
            + self._dataset.observable_names
        )
        design_names = self._dataset.design_variable_names
        data = self._dataset.get_view(
            variable_names=design_names + function_names
        ).to_numpy()
        som = self.__build_som_from_vars(data, som_grid_nx, som_grid_ny)
        bmu_coords = array([som.winner(datapoint) for datapoint in data])
        som_shape = som.get_weights().shape[:2]
        cluster_indices = array([x * som_shape[1] + y for x, y in bmu_coords])
        coord_2d_offset = self.__coord2d_to_coords_offsets(bmu_coords)
        self.materials_for_plotting["SOM"] = coord_2d_offset
        self._dataset.add_variable(
            variable_name="SOM_indx",
            data=cluster_indices,
            group_name=self._dataset.FUNCTION_GROUP,
        )
        self._dataset.add_variable(
            variable_name="SOM_i",
            data=bmu_coords[:, 0],
            group_name=self._dataset.FUNCTION_GROUP,
        )
        self._dataset.add_variable(
            variable_name="SOM_j",
            data=bmu_coords[:, 1],
            group_name=self._dataset.FUNCTION_GROUP,
        )
        self._dataset.add_variable(
            variable_name="SOM_x",
            data=coord_2d_offset[:, 0],
            group_name=self._dataset.FUNCTION_GROUP,
        )
        self._dataset.add_variable(
            variable_name="SOM_y",
            data=coord_2d_offset[:, 1],
            group_name=self._dataset.FUNCTION_GROUP,
        )

    @staticmethod
    def __coord2d_to_coords_offsets(
        som_coord: IntegerArray,
        max_offset: float = 0.6,
    ) -> RealArray:
        """Take a coord array from SOM and adds an offset.

        The offset is added to the coordinates of the
        elements in the cluster so that they can be distinguished at display.

        Args:
            som_coord: The SOM coordinates.
            max_offset: The maximum offset of the grid.

        Returns:
            The coordinates.
        """
        coord_2d = som_coord[:, :2]
        coord_2d_offset = array(coord_2d, dtype=float64)
        coord_indx = som_coord[:, -1]
        y_vars = bincount(coord_indx)
        i = nonzero(y_vars)[0]
        uniques_occ = array(list(zip(i, y_vars[i], strict=False)))
        unique_indx = uniques_occ[:, 0]
        max_occ = np_max(uniques_occ[:, 1])
        max_subarr_size = floor(sqrt(max_occ)) + 1
        dxdy_max = max_offset / (max_subarr_size - 1)
        for grp in unique_indx:
            inds_of_grp = (coord_indx == grp).nonzero()[0]
            subarr_size = sqrt(len(inds_of_grp))
            if floor(subarr_size) < subarr_size:
                subarr_size = floor(subarr_size) + 1
            else:
                subarr_size = floor(subarr_size)
            # Otherwise single individual then no need to build a grid
            if subarr_size > 1:
                grid = mgrid[0:subarr_size, 0:subarr_size] * dxdy_max
                gridx = grid[0, :, :].flatten()
                gridy = grid[1, :, :].flatten()
                for k, ind_in_grp in enumerate(inds_of_grp):
                    coord_2d_offset[ind_in_grp, 0] += gridx[k]
                    coord_2d_offset[ind_in_grp, 1] += gridy[k]
        return coord_2d_offset
