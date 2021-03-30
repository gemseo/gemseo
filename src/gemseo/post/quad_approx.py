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
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Quadratic approximations of functions from the optimization history
*******************************************************************
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from math import ceil
from os.path import splitext

import numpy as np
import pylab
from future import standard_library
from matplotlib import pyplot
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatter
from pylab import plt

from gemseo.post.core.colormaps import PARULA
from gemseo.post.core.hessians import SR1Approx
from gemseo.post.opt_post_processor import OptPostProcessor

standard_library.install_aliases()
from gemseo import LOGGER


class QuadApprox(OptPostProcessor):
    """
    The **QuadApprox** post processing
    performs a quadratic approximation of a given function
    from an optimization history
    and plot the results as cuts of the approximation.

    The function index can be passed as option.
    It is possible either to save the plot, to show the plot or both.
    """

    SR1_APPROX = "SR1"

    def __init__(self, opt_problem):
        """
        Constructor

        :param opt_problem : the optimization problem to run
        """
        super(QuadApprox, self).__init__(opt_problem)
        self.grad_opt = None

    def _plot(
        self,
        function,
        save=True,
        show=False,
        file_path="quad_approx",
        func_index=None,
        extension="pdf",
    ):
        """
        Builds the plot and saves it

        :param function: function name to build quadratic approximation
        :type function: str
        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param share_y: if True, all Y axis are the same,
            useful to compare sensitivities
        :type share_y: bool
        :param func_index: functional index
        :type func_index: int
        :param extension: file extension
        :type extension: str
        """
        b_mat = self.__build_approx(function, func_index)
        self.out_data_dict["b_mat"] = b_mat
        fig = self.__plot_hessian(b_mat, function)

        root = splitext(file_path)[0]
        file_path = root + "_hess_approx"
        self._save_and_show(
            fig, save=save, show=show, file_path=file_path, extension=extension
        )
        fig = self.__plot_variations(b_mat)

        file_path = root + "_quad_approx"
        self._save_and_show(
            fig, save=save, show=show, file_path=file_path, extension=extension
        )

    def __build_approx(self, function, func_index):
        """
        Builds the approximation

        :param method: SR1 or BFGS
        """
        apprx = SR1Approx(self.database)
        n_x = self.opt_problem.get_dimension()
        max_i = int(1.5 * n_x)
        # Avoid using alpha scaling for hessian otherwise diagonal is messy
        out = apprx.build_approximation(
            function, at_most_niter=max_i, return_x_grad=True, func_index=func_index
        )
        b_mat = out[0]
        grad_opt = out[-1]

        self.grad_opt = grad_opt
        return b_mat

    @staticmethod
    def __plot_hessian(hessian, function):
        """
        Plots the Hessian of the function

        :param hessian: the hessian
        """
        fig = pylab.figure()
        pylab.plt.xlabel(r"$x_i$", fontsize=16)
        pylab.plt.ylabel(r"$x_j$", fontsize=16)
        vmax = max(abs(np.max(hessian)), abs(np.min(hessian)))
        linthresh = 10 ** ((np.log10(vmax) - 5.0))
        # SymLog is a symetric log scale adapted to negative values
        pylab.imshow(
            hessian,
            cmap=PARULA,
            interpolation="nearest",
            norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
        )
        pylab.grid(True)
        l_f = LogFormatter(base=10, labelOnlyBase=False)
        thick_min = int(np.log10(linthresh))
        thick_max = int(np.log10(vmax))
        thick_num = thick_max - thick_min + 1
        lvls_pos = np.logspace(thick_min, thick_max, num=thick_num, base=10.0)
        levels_neg = np.sort(-lvls_pos)
        levels = np.concatenate((levels_neg, lvls_pos))
        pylab.plt.colorbar(ticks=levels, format=l_f)
        fig.suptitle("Hessian matrix SR1 approximation of " + str(function))
        return fig

    @staticmethod
    def unnormalize_vector(xn_array, ivar, lower_bounds, upper_bounds):
        """Unormalize a variable with respect to bounds

        :param xn_array: normalized variable (array)
        :param ivar: index of variable bound
        :param lower_bounds: param upper_bounds:
        :param upper_bounds:
        :returns: unnormalized array
        :rtype: numpy array

        """
        l_b = lower_bounds[ivar]
        u_b = upper_bounds[ivar]
        return (u_b - l_b) * xn_array * 0.5 + 0.5 * (l_b + u_b)

    def __plot_variations(self, hessian):
        """Plots the variation plots of the function wrt all variables """
        ndv = hessian.shape[0]
        ncols = int(np.sqrt(ndv)) + 1
        nrows = int(ceil(float(ndv) / ncols))

        xn_vars = np.arange(-1.0, 1.0, 0.01)
        lower_bounds = self.opt_problem.design_space.get_lower_bounds()
        upper_bounds = self.opt_problem.design_space.get_upper_bounds()
        fig = plt.figure()

        for i in range(ndv):
            ax_i = plt.subplot(nrows, ncols, i + 1)
            f_vals = xn_vars ** 2 * hessian[i, i] + self.grad_opt[i] * xn_vars
            self.out_data_dict[i] = f_vals

            x_vars = self.unnormalize_vector(xn_vars, i, lower_bounds, upper_bounds)
            ax_i.plot(x_vars, f_vals, "-", lw=2)

            start, stop = ax_i.get_xlim()
            ax_i.xaxis.set_ticks(np.arange(start, stop, 0.4999999 * (stop - start)))
            start, stop = ax_i.get_ylim()
            ax_i.yaxis.set_ticks(np.arange(start, stop, 0.24999999 * (stop - start)))
            ax_i.set_xlabel(r"$x_{" + str(i) + "}$", fontsize=14)
        pyplot.tight_layout()
        return fig
