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
"""Quadratic approximations of functions from the optimization history."""

from __future__ import division, unicode_literals

import logging
from math import ceil
from typing import Optional

import numpy as np
import pylab
from matplotlib import pyplot
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatter
from numpy import ndarray
from pylab import plt

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.core.colormaps import PARULA
from gemseo.post.core.hessians import SR1Approx
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.compatibility.matplotlib import SymLogNorm

LOGGER = logging.getLogger(__name__)


class QuadApprox(OptPostProcessor):
    """Quadratic approximation of a function.

    And cuts of the approximation.

    The function index can be passed as option.
    """

    DEFAULT_FIG_SIZE = (9.0, 6.0)

    SR1_APPROX = "SR1"

    def __init__(
        self,
        opt_problem,  # type: OptimizationProblem
    ):  # type: (...) -> None
        super(QuadApprox, self).__init__(opt_problem)
        self.grad_opt = None

    def _plot(
        self,
        function,  # type: str
        func_index=None,  # type: Optional[int]
    ):  # type: (...) -> None
        """Build the plot and save it.

        Args:
            function: The function name to build the quadratic approximation.
            func_index: The index of the output of interest
                to be defined if the function has a multidimensional output.
                If None and if the output is multidimensional, an error is raised.
        """
        b_mat = self.__build_approx(function, func_index)
        self.out_data_dict["b_mat"] = b_mat
        fig = self.__plot_hessian(b_mat, function)
        self._add_figure(fig, "hess_approx")
        fig = self.__plot_variations(b_mat)
        self._add_figure(fig, "quad_approx")

    def __build_approx(
        self,
        function,  # type: str
        func_index,  # type: int
    ):  # type: (...) -> ndarray
        """Build the approximation.

        Args:
            function: The function name to build the quadratic approximation.
            func_index: The index of the output of interest
                to be defined if the function has a multidimensional output.

        Returns:
             The approximation.
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

    @classmethod
    def __plot_hessian(
        cls,
        hessian,  # type: ndarray
        function,  # type: str
    ):  # type: (...) -> Figure
        """Plot the Hessian of the function.

        Args:
            hessian: The Hessian of the function.
            function: The function name.

        Returns:
            The plot of the Hessian of the function.
        """
        fig = pylab.figure(figsize=cls.DEFAULT_FIG_SIZE)
        pylab.plt.xlabel(r"$x_i$", fontsize=16)
        pylab.plt.ylabel(r"$x_j$", fontsize=16)
        vmax = max(abs(np.max(hessian)), abs(np.min(hessian)))
        linthresh = 10 ** (np.log10(vmax) - 5.0)

        norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)

        # SymLog is a symmetric log scale adapted to negative values
        pylab.imshow(
            hessian,
            cmap=PARULA,
            interpolation="nearest",
            norm=norm,
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
    def unnormalize_vector(
        xn_array,  # type: ndarray
        ivar,  # type: int
        lower_bounds,  # type: ndarray
        upper_bounds,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Unormalize a variable with respect to bounds.

        Args:
            xn_array: The normalized variable.
            ivar: The index of the variable bound.
            lower_bounds: The lower bounds of the variable.
            upper_bounds: The upper bounds of the variable.

        Returns:
            The unnormalized variable.
        """
        l_b = lower_bounds[ivar]
        u_b = upper_bounds[ivar]
        return (u_b - l_b) * xn_array * 0.5 + 0.5 * (l_b + u_b)

    def __plot_variations(
        self,
        hessian,  # type: ndarray
    ):  # type: (...) -> Figure
        """Plot the variation plot of the function w.r.t. all variables.

        Args:
            hessian: The Hessian of the function.

        Returns:
            The plot of the variation of the function w.r.t. all variables.
        """
        ndv = hessian.shape[0]
        ncols = int(np.sqrt(ndv)) + 1
        nrows = int(ceil(float(ndv) / ncols))

        xn_vars = np.arange(-1.0, 1.0, 0.01)
        lower_bounds = self.opt_problem.design_space.get_lower_bounds()
        upper_bounds = self.opt_problem.design_space.get_upper_bounds()
        fig = plt.figure(figsize=self.DEFAULT_FIG_SIZE)

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
