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
Box plots to quantify optimum robustness
****************************************
"""
from __future__ import absolute_import, division, unicode_literals

from math import sqrt

import matplotlib.pyplot as plt
import numpy.random as npr
from future import standard_library
from numpy import zeros

from gemseo.post.core.robustness_quantifier import RobustnessQuantifier
from gemseo.post.opt_post_processor import OptPostProcessor

standard_library.install_aliases()
from gemseo import LOGGER


class Robustness(OptPostProcessor):
    """
    The **Robustness** post processing
    performs a quadratic approximation from an optimization history,
    and plot the results as cuts of the approximation
    computes the quadratic approximations of all the output functions,
    propagate analytically a normal distribution centered on the optimal
    design variable with a standard deviation which is a percentage
    of the mean passed in option (default: 1%)
    and plot the corresponding output boxplot.

    It is possible either to save the plot, to show the plot or both.
    """

    SR1_APPROX = "SR1"

    def _plot(
        self, save=True, show=False, stddev=0.01, file_path="boxplot", extension="pdf"
    ):
        """
        Builds the plot and saves it

        :param function: function name to build quadratic approximation
        :type function: bool
        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param stddev: standard deviation of inputs as fraction of x bounds
        :type stddev: float
        :param share_y: if True, all Y axis are the same,
            useful to comare sensitivities
        :type share_y: bool
        :param extension: file extension
        :type extension: str
        """
        fig = self.__boxplot(stddev)
        self._save_and_show(
            fig, save=save, show=show, file_path=file_path, extension=extension
        )

    def __boxplot(self, stddev=0.01):
        """
        Plots the Hessian of the function

        :param stddev : standard deviation of inputs as fraction of x bounds
        """
        robustness = RobustnessQuantifier(self.database, "SR1")
        n_x = self.opt_problem.get_dimension()
        cov = zeros((n_x, n_x))
        upper_bounds = self.opt_problem.design_space.get_upper_bounds()
        lower_bounds = self.opt_problem.design_space.get_lower_bounds()
        bounds_range = upper_bounds - lower_bounds
        cov[list(range(n_x)), list(range(n_x))] = (stddev * (bounds_range)) ** 2

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
                    data.append(npr.normal(loc=mean, scale=sqrt(var), size=500))
                    legend = func_name
                    if dim > 1:
                        legend += "_" + str(i + 1)
                    funcs_names.append(legend)

        fig = plt.figure()
        fig.suptitle(
            "Box plot of the optimization functions "
            + "with normalized stddev "
            + str(stddev)
        )
        plt.boxplot(data, showfliers=False, labels=funcs_names)

        return fig
