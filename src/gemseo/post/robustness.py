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
"""Boxplots to quantify the robustness of the optimum."""
from __future__ import annotations

import logging
from math import sqrt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import zeros
from numpy.random import normal
from numpy.random import seed

from gemseo.post.core.robustness_quantifier import RobustnessQuantifier
from gemseo.post.opt_post_processor import OptPostProcessor

LOGGER = logging.getLogger(__name__)


class Robustness(OptPostProcessor):
    """Uncertainty quantification at the optimum.

    Compute the quadratic approximations of all the output functions,
    propagate analytically a normal distribution centered
    on the optimal design variables
    with a standard deviation which is a percentage of the mean passed in option
    (default: 1%)
    and plot the corresponding output boxplot.
    """

    DEFAULT_FIG_SIZE = (8.0, 5.0)

    SR1_APPROX = "SR1"

    def _plot(
        self,
        stddev: float = 0.01,
    ) -> None:
        """
        Args:
            stddev: The standard deviation of the normal uncertain variable
                to be added to the optimal design value;
                expressed as a fraction of the bounds of the design variables.
        """  # noqa: D205, D212, D415
        seed(0)
        self._add_figure(self.__boxplot(stddev))

    def __boxplot(
        self,
        standard_deviation: float = 0.01,
    ) -> Figure:
        """Plot the Hessian of the function.

        Args:
            standard_deviation: The standard deviation of the normal uncertain variable
                to be added to the optimal design value;
                expressed as a fraction of the bounds of the design variables.

        Returns:
            A plot of the Hessian of the function.
        """
        problem = self.opt_problem
        design_space = problem.design_space
        bounds_range = design_space.get_upper_bounds() - design_space.get_lower_bounds()
        n_x = problem.get_dimension()
        cov = zeros((n_x, n_x))
        cov[range(n_x), range(n_x)] = (standard_deviation * bounds_range) ** 2

        robustness = RobustnessQuantifier(self.database)
        function_samples = []
        function_names = []
        for func in self.opt_problem.get_all_functions():
            func_name = database_func_name = func.name
            if self._change_obj and func_name == self._neg_obj_name:
                func_name = self._obj_name

            dim = func.dim
            at_most_niter = int(1.5 * n_x)
            for func_index in range(dim):
                robustness.compute_approximation(
                    funcname=database_func_name,
                    at_most_niter=at_most_niter,
                    func_index=func_index,
                    b0_mat=zeros((n_x, n_x)),
                )
                x_ref = robustness.x_ref
                mean = robustness.compute_expected_value(x_ref, cov)
                if self._change_obj:
                    mean = -mean

                variance = robustness.compute_variance(x_ref, cov)
                if variance > 0:  # Otherwise normal doesn't work
                    function_samples.append(
                        normal(loc=mean, scale=sqrt(variance), size=500)
                    )
                    legend = func_name
                    if dim > 1:
                        legend += f" ({func_index})"
                    function_names.append(legend)

        fig = plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        fig.suptitle(
            "Boxplot of the optimization functions "
            f"with normalized stddev {standard_deviation}"
        )
        plt.boxplot(function_samples, showfliers=False, labels=function_names)
        return fig
