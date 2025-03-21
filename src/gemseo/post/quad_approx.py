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

from math import ceil
from typing import TYPE_CHECKING
from typing import ClassVar

import numpy as np
from matplotlib import pyplot
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import SymmetricalLogLocator
from numpy import arange

from gemseo.post.base_post import BasePost
from gemseo.post.core.colormaps import PARULA
from gemseo.post.core.hessians import SR1Approx
from gemseo.post.quad_approx_settings import QuadApprox_Settings

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from gemseo.typing import NumberArray


class QuadApprox(BasePost[QuadApprox_Settings]):
    """Quadratic approximation of a function.

    And cuts of the approximation.

    The function index can be passed as option.
    """

    Settings: ClassVar[type[QuadApprox_Settings]] = QuadApprox_Settings

    def _plot(self, settings: QuadApprox_Settings) -> None:
        function = settings.function
        func_index = settings.func_index

        problem = self.optimization_problem
        if function == self._obj_name:
            b_mat, grad_opt = self.__build_approx(
                self._standardized_obj_name, func_index
            )
            if not (problem.minimize_objective or problem.use_standardized_objective):
                grad_opt *= -1
                b_mat *= -1
                function = self._standardized_obj_name
        else:
            if function in problem.constraints.original_to_current_names:
                function = problem.constraints.original_to_current_names[function][0]

            b_mat, grad_opt = self.__build_approx(function, func_index)

        self.materials_for_plotting["b_mat"] = b_mat
        self._add_figure(
            self.__plot_hessian(b_mat, function, settings.fig_size), "hess_approx"
        )
        self._add_figure(
            self.__plot_variations(b_mat, settings.fig_size, grad_opt),
            "quad_approx",
        )

    def __build_approx(
        self,
        function: str,
        func_index: int | None,
    ) -> tuple[NumberArray, NumberArray]:
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
        b_mat, _, _, grad_opt = SR1Approx(self.database).build_approximation(
            function,
            at_most_niter=int(1.5 * self.optimization_problem.design_space.dimension),
            return_x_grad=True,
            func_index=func_index,
        )
        assert grad_opt is not None
        return b_mat, grad_opt

    def __plot_hessian(
        self,
        hessian: NumberArray,
        function: str,
        fig_size: tuple[float, float],
    ) -> Figure:
        """Plot the Hessian of the function.

        Args:
            hessian: The Hessian of the function.
            function: The function name.

        Returns:
            The plot of the Hessian of the function.
        """
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        vmax = max(abs(np.max(hessian)), abs(np.min(hessian)))
        linear_threshold = 10 ** (np.log10(vmax) - 5.0)

        # SymLog is a symmetric log scale adapted to negative values
        img = ax.imshow(
            hessian,
            cmap=PARULA,
            interpolation="nearest",
            norm=SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linear_threshold),
        )
        ticks = arange(self.optimization_problem.design_space.dimension)
        design_variable_names = self._get_design_variable_names(simplify=True)
        ax.set_xticks(ticks)
        ax.set_xticklabels(design_variable_names, rotation=45)
        ax.set_yticks(ticks)
        ax.set_yticklabels(design_variable_names)
        fig.colorbar(
            img,
            ticks=SymmetricalLogLocator(linthresh=linear_threshold, base=10),
            format=LogFormatterSciNotation(),
        )
        fig.suptitle(f"Hessian matrix SR1 approximation of {function}")
        fig.tight_layout()
        return fig

    @staticmethod
    def unnormalize_vector(
        xn_array: NumberArray,
        ivar: int,
        lower_bounds: NumberArray,
        upper_bounds: NumberArray,
    ) -> NumberArray:
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
        hessian: NumberArray,
        fig_size: tuple[float, float],
        grad_opt: NumberArray,
    ) -> Figure:
        """Plot the variation plot of the function w.r.t. all variables.

        Args:
            hessian: The Hessian of the function.

        Returns:
            The plot of the variation of the function w.r.t. all variables.
        """
        ndv = hessian.shape[0]
        ncols = int(np.sqrt(ndv)) + 1
        nrows = ceil(float(ndv) / ncols)

        xn_vars = np.arange(-1.0, 1.0, 0.01)
        lower_bounds = self.optimization_problem.design_space.get_lower_bounds()
        upper_bounds = self.optimization_problem.design_space.get_upper_bounds()
        fig = plt.figure(figsize=fig_size)

        for i, design_variable_name in enumerate(self._get_design_variable_names()):
            ax_i = plt.subplot(nrows, ncols, i + 1)
            f_vals = xn_vars**2 * hessian[i, i] + grad_opt[i] * xn_vars
            self.materials_for_plotting[i] = f_vals

            x_vars = self.unnormalize_vector(xn_vars, i, lower_bounds, upper_bounds)
            ax_i.plot(x_vars, f_vals, "-", lw=2)

            start, stop = ax_i.get_xlim()
            ax_i.xaxis.set_ticks(np.arange(start, stop, 0.4999999 * (stop - start)))
            start, stop = ax_i.get_ylim()
            ax_i.yaxis.set_ticks(np.arange(start, stop, 0.24999999 * (stop - start)))
            ax_i.set_xlabel(design_variable_name)

        pyplot.tight_layout()
        return fig
