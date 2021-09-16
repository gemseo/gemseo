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
"""Box plots to quantify optimum robustness."""
from __future__ import division, unicode_literals

import logging
from math import sqrt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import zeros
from numpy.random import normal

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
        stddev=0.01,  # type: float
    ):  # type: (...) -> None
        """
        Args:
            stddev: The standard deviation of the inputs as fraction of x bounds.
        """
        self._add_figure(self.__boxplot(stddev))

    def __boxplot(
        self,
        stddev=0.01,  # type: float
    ):  # type: (...) -> Figure
        """Plot the Hessian of the function.

        Args:
            stddev: The standard deviation of the inputs as fraction of x bounds.

        Returns:
            A plot of the Hessian of the function.
        """
        robustness = RobustnessQuantifier(self.database, "SR1")
        n_x = self.opt_problem.get_dimension()
        cov = zeros((n_x, n_x))
        upper_bounds = self.opt_problem.design_space.get_upper_bounds()
        lower_bounds = self.opt_problem.design_space.get_lower_bounds()
        bounds_range = upper_bounds - lower_bounds
        cov[list(range(n_x)), list(range(n_x))] = (stddev * bounds_range) ** 2

        data = []
        funcs_names = []
        for func in self.opt_problem.get_all_functions():
            func_name = func.name
            dim = func.dim
            for i in range(dim):
                b0_mat = zeros((n_x, n_x))
                robustness.compute_approximation(
                    funcname=func_name,
                    at_most_niter=int(1.5 * n_x),
                    func_index=i,
                    b0_mat=b0_mat,
                )
                x_ref = robustness.x_ref
                mean = robustness.compute_expected_value(x_ref, cov)
                var = robustness.compute_variance(x_ref, cov)
                if var > 0:  # Otherwise normal doesnt work
                    data.append(normal(loc=mean, scale=sqrt(var), size=500))
                    legend = func_name
                    if dim > 1:
                        legend += "_" + str(i + 1)
                    funcs_names.append(legend)

        fig = plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        fig.suptitle(
            "Box plot of the optimization functions "
            "with normalized stddev {}".format(stddev)
        )
        plt.boxplot(data, showfliers=False, labels=funcs_names)

        return fig
