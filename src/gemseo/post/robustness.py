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

from math import sqrt
from typing import ClassVar

import matplotlib.pyplot as plt
from numpy import zeros
from numpy.random import default_rng

from gemseo import SEED
from gemseo.post.base_post import BasePost
from gemseo.post.core.robustness_quantifier import RobustnessQuantifier
from gemseo.post.robustness_settings import RobustnessSettings
from gemseo.utils.string_tools import repr_variable

LOGGER = logging.getLogger(__name__)


class Robustness(BasePost[RobustnessSettings]):
    """Uncertainty quantification at the optimum.

    Compute the quadratic approximations of all the output functions, propagate
    analytically a normal distribution centered on the optimal design variables with a
    standard deviation which is a percentage of the mean passed in option (default: 1%)
    and plot the corresponding output boxplot.
    """

    SR1_APPROX: ClassVar[str] = "SR1"
    Settings: ClassVar[type[RobustnessSettings]] = RobustnessSettings

    def _plot(self, settings: RobustnessSettings) -> None:
        standard_deviation = settings.stddev
        problem = self.optimization_problem
        design_space = problem.design_space
        bounds_range = design_space.get_upper_bounds() - design_space.get_lower_bounds()
        n_x = problem.design_space.dimension
        cov = zeros((n_x, n_x))
        cov[range(n_x), range(n_x)] = (standard_deviation * bounds_range) ** 2

        robustness = RobustnessQuantifier(self.database)
        function_samples = []
        function_names = []
        for func in self.optimization_problem.functions:
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
                        default_rng(SEED).normal(mean, sqrt(variance), 500)
                    )
                    function_names.append(repr_variable(func_name, func_index, dim))

        fig = plt.figure(figsize=settings.fig_size)
        fig.suptitle(
            "Boxplot of the optimization functions "
            f"with normalized stddev {standard_deviation}"
        )
        plt.boxplot(function_samples, showfliers=False, labels=function_names)
