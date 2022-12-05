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
from __future__ import annotations

import logging
from math import ceil

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

    def __init__(  # noqa:D107
        self,
        opt_problem: OptimizationProblem,
    ) -> None:
        super().__init__(opt_problem)
        self.grad_opt = None

    def _plot(
        self,
        function: str,
        func_index: int | None = None,
    ) -> None:
        """Build the plot and save it.

        Args:
            function: The function name to build the quadratic approximation.
            func_index: The index of the output of interest
                to be defined if the function has a multidimensional output.
                If ``None`` and if the output is multidimensional, an error is raised.
        """  # noqa: D205, D212, D415
        problem = self.opt_problem
        if function == self._obj_name:
            b_mat = self.__build_approx(self._standardized_obj_name, func_index)
            if not (problem.minimize_objective or problem.use_standardized_objective):
                self.grad_opt *= -1
                b_mat *= -1
                function = self._standardized_obj_name
        else:
            if function in problem.constraint_names:
                function = problem.constraint_names[function][0]

            b_mat = self.__build_approx(function, func_index)

        self.materials_for_plotting["b_mat"] = b_mat
        self._add_figure(self.__plot_hessian(b_mat, function), "hess_approx")
        self._add_figure(self.__plot_variations(b_mat), "quad_approx")

    def __build_approx(
        self,
        function: str,
        func_index: int | None,
    ) -> ndarray:
        """Build the approximation.

        Args:
            function: The function name to build the quadratic approximation.
            func_index: The index of the output of interest
                to be defined if the function has a multidimensional output.
                If ``None`` and if the output is multidimensional, an error is raised.

        Returns:
             The approximation.
        """
        # Avoid using alpha scaling for hessian otherwise diagonal is messy
        b_mat, _, _, self.grad_opt = SR1Approx(self.database).build_approximation(
            function,
            at_most_niter=int(1.5 * self.opt_problem.get_dimension()),
            return_x_grad=True,
            func_index=func_index,
        )
        return b_mat

    @classmethod
    def __plot_hessian(
        cls,
        hessian: ndarray,
        function: str,
    ) -> Figure:
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
        linear_threshold = 10 ** (np.log10(vmax) - 5.0)

        # SymLog is a symmetric log scale adapted to negative values
        pylab.imshow(
            hessian,
            cmap=PARULA,
            interpolation="nearest",
            norm=SymLogNorm(linthresh=linear_threshold, vmin=-vmax, vmax=vmax),
        )
        pylab.grid(True)
        thick_min, thick_max = int(np.log10(linear_threshold)), int(np.log10(vmax))
        positive_levels = np.logspace(
            thick_min, thick_max, num=thick_max - thick_min + 1
        )
        pylab.plt.colorbar(
            ticks=np.concatenate((np.sort(-positive_levels), positive_levels)),
            format=LogFormatter(base=10),
        )
        fig.suptitle(f"Hessian matrix SR1 approximation of {function}")
        return fig

    @staticmethod
    def unnormalize_vector(
        xn_array: ndarray,
        ivar: int,
        lower_bounds: ndarray,
        upper_bounds: ndarray,
    ) -> ndarray:
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
        hessian: ndarray,
    ) -> Figure:
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
            f_vals = xn_vars**2 * hessian[i, i] + self.grad_opt[i] * xn_vars
            self.materials_for_plotting[i] = f_vals

            x_vars = self.unnormalize_vector(xn_vars, i, lower_bounds, upper_bounds)
            ax_i.plot(x_vars, f_vals, "-", lw=2)

            start, stop = ax_i.get_xlim()
            ax_i.xaxis.set_ticks(np.arange(start, stop, 0.4999999 * (stop - start)))
            start, stop = ax_i.get_ylim()
            ax_i.yaxis.set_ticks(np.arange(start, stop, 0.24999999 * (stop - start)))
            ax_i.set_xlabel(r"$x_{" + str(i) + "}$", fontsize=14)
        pyplot.tight_layout()
        return fig
