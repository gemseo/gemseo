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

import matplotlib
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
from numpy import where
from numpy import zeros
from pylab import plt

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.core.colormaps import PARULA
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.third_party import sompy

LOGGER = logging.getLogger(__name__)


class SOM(OptPostProcessor):
    """Self organizing map clustering optimization history.

    Options of the plot method are the x- and y- numbers of cells in the SOM.
    """

    DEFAULT_FIG_SIZE = (12.0, 18.0)

    def __init__(  # noqa:D107
        self,
        opt_problem: OptimizationProblem,
    ) -> None:
        super().__init__(opt_problem)
        self.som = None
        self.cmap = PARULA

    @staticmethod
    def __build_som_from_vars(
        x_vars: ndarray,
        som_grid_nx: int = 5,
        som_grid_ny: int = 5,
        initmethod: str = "pca",
        verbose: str = "off",
    ) -> SOM:
        """Builds the SOM from the design variables history.

        Args:
            x_vars: The design variables history (n_iter,n_dv).
            som_grid_nx: The number of neurons in the x direction.
            som_grid_ny: The number of neurons in the y direction.
            initmethod: The initialization method for the SOM.
            verbose: The verbose for SOM training.

        Returns:
            The self organizing map
        """
        LOGGER.info("Building Self Organizing Map from optimization history:")
        LOGGER.info("    Number of neurons in x direction = %s", str(som_grid_nx))
        LOGGER.info("    Number of neurons in y direction = %s", str(som_grid_ny))
        var_som = sompy.SOM(
            "som",
            x_vars,
            mapsize=[som_grid_ny + 1, som_grid_nx + 1],
            initmethod=initmethod,
        )
        var_som.init_map()
        var_som.train(verbose=verbose)
        return var_som

    def _plot(
        self,
        n_x: int = 4,
        n_y: int = 4,
        annotate: bool = False,
    ) -> None:
        """
        Args:
            n_x: The number of grids in x.
            n_y: The number of grids in y.
            annotate: If True, add label of neuron value to SOM plot.
        """  # noqa: D205, D212, D415
        criteria = [
            self.opt_problem.get_objective_name()
        ] + self.opt_problem.get_constraints_names()
        all_data = self.database.get_all_data_names()
        # Ensure that the data is available in the database
        for criterion in criteria:
            if criterion not in all_data:
                criteria.remove(criterion)
        figure = plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        figure.suptitle("Self Organizing Maps of the design space", fontsize=14)
        subplot_number = 0
        self.__compute(n_x, n_y)
        for criterion in criteria:
            f_hist, _ = self.database.get_complete_history(
                ["SOM_i", "SOM_j", "SOM_indx", criterion]
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
            f_hist, _ = self.database.get_complete_history(
                ["SOM_i", "SOM_j", "SOM_indx", criterion]
            )
            if isinstance(f_hist[0][3], ndarray):
                for k in range(f_hist[0][3].size):
                    self.__plot_som_from_scalar_data(
                        [f_h[0:3] + [f_h[3][k]] for f_h in f_hist],
                        f"{criterion}_{k}",
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

        self._add_figure(figure)

    def __plot_som_from_scalar_data(
        self,
        f_hist_scalar: ndarray,
        criteria: str,
        fig_indx: int,
        grid_size_x: int = 3,
        grid_size_y: int = 20,
        annotate: bool = False,
    ):
        """Builds the SOM plot after computation for a given criteria.

        Args:
            criteria: The criteria to show.
            f_hist_scalar: The scalar data to show.
            fig_indx: The axe index in the figure.
            grid_size_x: The number of SOMs in the grid on the x axis.
            grid_size_y: The number of SOMs in the grid on the y axis.
            annotate: If True, add label with average value of neural.
        """
        f_hist = array(f_hist_scalar).T.real
        unique_ind = unique(f_hist[2, :])
        average = {}
        for _, som_id in enumerate(unique_ind):
            where_somid = where(f_hist[2, :] == som_id)[0]
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
        axe = plt.subplot(grid_size_y, grid_size_x, fig_indx)
        minv = np_min(mat_ij[non_empty])
        maxv = np_max(mat_ij[non_empty])
        self.materials_for_plotting[fig_indx] = mat_ij
        im1 = axe.imshow(
            mat_ij,
            vmin=minv - 0.01 * abs(minv),
            vmax=maxv + 0.01 * abs(maxv),
            cmap=self.cmap,
            interpolation="nearest",
            aspect="auto",
        )  # "spectral" "hot" "RdBu_r"

        if annotate:
            crit_format = "%1.2g"
            for i in range(mat_ij.shape[0]):
                for j in range(mat_ij.shape[0]):
                    _ = axe.text(
                        j,
                        i,
                        crit_format % mat_ij[i, j],
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=7,
                    )

        axe.set_title(criteria, fontsize=12)
        cax, kwa = matplotlib.colorbar.make_axes([axe])
        plt.colorbar(im1, cax=cax, **kwa)
        im1.axes.get_xaxis().set_visible(False)
        im1.axes.get_yaxis().set_visible(False)
        return axe

    def __compute(
        self,
        som_grid_nx: int = 5,
        som_grid_ny: int = 5,
    ):
        """Build the SOM from optimization history.

        Args:
            som_grid_nx: The number of neurons in the x direction.
            som_grid_ny: The number of neurons in the y direction.
        """
        x_history = self.database.get_x_history()
        x_vars = array(x_history).real
        self.som = self.__build_som_from_vars(x_vars, som_grid_nx, som_grid_ny)
        som_cluster_index = self.som.project_data(x_vars)
        som_coord = array(self.som.ind_to_xy(som_cluster_index), dtype=int32)
        coord_2d_offset = self.__coord2d_to_coords_offsets(som_coord)
        self.materials_for_plotting["SOM"] = coord_2d_offset
        for i, x_vars in enumerate(x_history):
            self.database.store(
                x_vars,
                {
                    "SOM_indx": som_cluster_index[i],
                    "SOM_i": som_coord[i, 0],
                    "SOM_j": som_coord[i, 1],
                    "SOM_x": coord_2d_offset[i, 0],
                    "SOM_y": coord_2d_offset[i, 1],
                },
            )

    @staticmethod
    def __coord2d_to_coords_offsets(
        som_coord: ndarray,
        max_ofset: float = 0.6,
    ) -> ndarray:
        """Take a coord array from SOM and adds an offset.

        The offset is added to the coordinates of the
        elements in the cluster so that they can be distinguished at display.

        Args:
            som_coord: The SOM coordinates.
            max_ofset: The maximum offset of the grid.

        Returns:
            The coordinates.
        """
        coord_2d = som_coord[:, :2]
        coord_2d_offset = array(coord_2d, dtype=float64)
        coord_indx = som_coord[:, -1]
        y_vars = bincount(coord_indx)
        i = nonzero(y_vars)[0]
        uniques_occ = array(list(zip(i, y_vars[i])))
        unique_indx = uniques_occ[:, 0]
        max_occ = np_max(uniques_occ[:, 1])
        max_subarr_size = floor(sqrt(max_occ)) + 1
        dxdy_max = max_ofset / (max_subarr_size - 1)
        for grp in unique_indx:
            inds_of_grp = where(coord_indx == grp)[0]
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
