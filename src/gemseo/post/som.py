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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Self Organizing Maps plots to display high dimensional design spaces
********************************************************************
"""

from __future__ import absolute_import, division, unicode_literals

from math import floor, sqrt

import matplotlib
from future import standard_library
from numpy import array, bincount, float64, int32, isnan, logical_not
from numpy import max as np_max
from numpy import mean, mgrid
from numpy import min as np_min
from numpy import ndarray, nonzero, unique, where, zeros
from pylab import plt

from gemseo.post.core.colormaps import PARULA
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.third_party.sompy import SOM as spy_som

standard_library.install_aliases()
from gemseo import LOGGER


class SOM(OptPostProcessor):
    """
    The **SOM** post processing
    perform a self organizing map
    clustering on optimization history

    Options of the plot method are the figure width and height,
    and the x- and y- number of cells in the SOM.
    It is also possible either to save the plot, to show the plot or both.
    """

    def __init__(self, opt_problem):
        """
        Constructor

        :param opt_problem : the optimization problem to run
        """
        super(SOM, self).__init__(opt_problem)
        self.som = None
        self.cmap = PARULA

    def _run(
        self,
        n_x=4,
        n_y=4,
        save=False,
        show=False,
        file_path="SOM",
        annotate=False,
        width=12,
        height=18,
        extension="pdf",
    ):
        """Computes the clustering

        :param n_x: x-size
        :type n_x: int
        :param n_y: y-size
        :type n_y: int
        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param annotate: add label of neuron value to SOM plot
        :param width: figure width
        :param height: figure height
        :param extension: file extension
        :type extension: str
        """

        criteria = [
            self.opt_problem.get_objective_name()
        ] + self.opt_problem.get_constraints_names()
        all_data = self.database.get_all_data_names()
        # Ensure that the data is available in the database
        for crit in criteria:
            if crit not in all_data:
                criteria.remove(crit)
        figure = self._plot(
            criteria, n_x, n_y, annotate=annotate, width=width, height=height
        )
        self._save_and_show(
            figure, save=save, show=show, file_path=file_path, extension=extension
        )

    @staticmethod
    def __build_som_from_vars(
        x_vars, som_grid_nx=5, som_grid_ny=5, initmethod="pca", verbose="off"
    ):
        """
        Builds the SOM from the design variables history

        :param x_vars:  the design variables history numpy array (n_iter,n_dv)
        :param som_grid_nx: number of neurons in the x direction
        :param som_grid_ny: number of neurons in the y direction
        :param initmethod: initialization method for the SOM
        :param verbose: verbose for SOM training
        """

        LOGGER.info("Building Self Organizing Map from optimization history:")
        LOGGER.info("    Number of neurons in x direction = %s", str(som_grid_nx))
        LOGGER.info("    Number of neurons in y direction = %s", str(som_grid_ny))
        var_som = spy_som(
            "som",
            x_vars,
            mapsize=[som_grid_ny + 1, som_grid_nx + 1],
            norm_method="var",
            initmethod=initmethod,
        )
        var_som.init_map()
        var_som.train(n_job=1, shared_memory="no", verbose=verbose)
        return var_som

    def _plot(self, criteria_list, n_x, n_y, width=12, height=18, annotate=False):
        """
        Shows the SOM view after computation for a given criteria list

        :param criteria_list: the criteria to show
        :param n_x: number of grids in x
        :param n_y: number of grids in y
        :param annotate: add label of neuron value to SOM plot
        """
        figure = plt.figure(figsize=(width, height), dpi=80)
        figure.suptitle("Self Organizing Maps of the design space", fontsize=14)
        subplot_number = 0
        self.__compute(n_x, n_y)
        for criteria in criteria_list:
            f_hist, _ = self.database.get_complete_history(
                ["SOM_i", "SOM_j", "SOM_indx", criteria]
            )
            if isinstance(f_hist[0][3], ndarray):
                dim_val = f_hist[0][3].size
                for k in range(dim_val):
                    subplot_number += 1

            else:
                subplot_number += 1

        grid_size_x = 3
        grid_size_y = subplot_number // grid_size_x
        if (subplot_number % grid_size_x) > 0:
            grid_size_y += 1

        fig_indx = 1
        for criteria in criteria_list:
            f_hist, _ = self.database.get_complete_history(
                ["SOM_i", "SOM_j", "SOM_indx", criteria]
            )
            if isinstance(f_hist[0][3], ndarray):
                dim_val = f_hist[0][3].size
                for k in range(dim_val):
                    f_hist_scalar = []
                    for f_h in f_hist:
                        scal_list = f_h[0:3]
                        scal_list.append(f_h[3][k])
                        f_hist_scalar.append(scal_list)
                    criteria_name = criteria + "_" + str(k)
                    self.__plot_som_from_scalar_data(
                        f_hist_scalar,
                        criteria_name,
                        fig_indx,
                        grid_size_x=grid_size_x,
                        grid_size_y=grid_size_y,
                        annotate=annotate,
                    )
                    fig_indx += 1

            else:
                self.__plot_som_from_scalar_data(
                    f_hist,
                    criteria,
                    fig_indx,
                    grid_size_x=grid_size_x,
                    grid_size_y=grid_size_y,
                    annotate=annotate,
                )
                fig_indx += 1
        return figure

    def __plot_som_from_scalar_data(
        self,
        f_hist_scalar,
        criteria,
        fig_indx,
        grid_size_x=3,
        grid_size_y=20,
        annotate=False,
    ):
        """
        Builds the SOM plot after computation for a given criteria

        :param criteria: the criteria to show
        :param f_hist_scalar: the scalar data to show
        :param fig_indx: the axe index in the figure
        :param grid_size_x: number of SOMs in the grid on the x axis
        :param grid_size_y: number of SOMs in the grid on the y axis
        :param annotate: add label with average value of neural
        """
        f_hist = array(f_hist_scalar).T.real
        unique_ind = unique(f_hist[2, :])
        average = {}
        for i, som_id in enumerate(unique_ind):
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
        self.out_data_dict[fig_indx] = mat_ij
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

    def __compute(self, som_grid_nx=5, som_grid_ny=5):
        """
        Builds the SOM from optimization history

        :param som_grid_nx: number of neurons in the x direction
        :param som_grid_ny: number of neurons in the y direction
        """
        x_history = self.database.get_x_history()
        x_vars = array(x_history).real
        self.som = self.__build_som_from_vars(x_vars, som_grid_nx, som_grid_ny)
        som_cluster_index = self.som.project_data(x_vars)
        som_coord = array(self.som.ind_to_xy(som_cluster_index), dtype=int32)
        coord_2d_offset = self.__coord2d_to_coords_offsets(som_coord)
        self.out_data_dict["SOM"] = coord_2d_offset
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
    def __coord2d_to_coords_offsets(som_coord, max_ofset=0.6):
        """
        Takes a coord array from SOM and adds an offset to the coordinates of
        the elements in the cluster so that they can be distinguished
        at display

        :param som_coord: the SOM coords array
        :paramtype som_coord: ndarray
        :param max_ofset: the maximum offset of the grid
        :paramtype: max_ofset: float
        :returns: a coordinate array
        :rtype: ndarray
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
