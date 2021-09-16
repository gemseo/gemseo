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
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Plot the partial sensitivity of the functions."""
from __future__ import division, unicode_literals

import logging
from typing import Mapping, Tuple

from matplotlib import pyplot
from matplotlib.figure import Figure
from numpy import absolute, argsort, array, atleast_2d, ndarray, savetxt, stack

from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.py23_compat import PY2

LOGGER = logging.getLogger(__name__)


class VariableInfluence(OptPostProcessor):
    """First order variable influence analysis.

    This post-processing computes df/dxi * (xi* - xi0)
    where xi0 is the initial value of the variable
    and xi* is the optimal value of the variable.

    Options of the plot method are the quantile level, the use of a
    logarithmic scale and the possibility to save the influent variables
    indices as a NumPy file.
    """

    DEFAULT_FIG_SIZE = (20.0, 5.0)

    def _plot(
        self,
        quantile=0.99,  # type: float
        absolute_value=False,  # type: bool
        log_scale=False,  # type: bool
        save_var_files=False,  # type: bool
    ):  # type: (...) -> None
        """
        Args:
            quantile: Between 0 and  1, the proportion of the total
                sensitivity to use as a threshold to filter the variables.
            absolute_value: If True, plot the absolute value of the influence.
            log_scale: If True, use a logarithmic scale.
            save_var_files: If True, save the influent variables indices as a NumPy file.
        """
        all_funcs = self.opt_problem.get_all_functions_names()
        _, x_opt, _, _, _ = self.opt_problem.get_optimum()
        x_0 = self.database.get_x_by_iter(0)
        if log_scale:
            absolute_value = True
        sens_dict = {}
        for func in all_funcs:
            func_gradient = self.database.get_gradient_name(func)
            grad = self.database.get_f_of_x(func_gradient, x_0)
            f_0 = self.database.get_f_of_x(func, x_0)
            f_opt = self.database.get_f_of_x(func, x_opt)
            if grad is not None:
                if len(grad.shape) == 1:
                    sens = grad * (x_opt - x_0)
                    delta_corr = (f_opt - f_0) / sens.sum()
                    sens *= delta_corr
                    if absolute_value:
                        sens = absolute(sens)
                    sens_dict[func] = sens
                else:
                    n_f, _ = grad.shape
                    for i in range(n_f):
                        sens = grad[i, :] * (x_opt - x_0)
                        delta_corr = (f_opt - f_0)[i] / sens.sum()
                        sens *= delta_corr
                        if absolute_value:
                            sens = absolute(sens)
                        sens_dict["{}_{}".format(func, i)] = sens

        fig = self.__generate_subplots(sens_dict, quantile, log_scale, save_var_files)

        self._add_figure(fig)

    def __get_quantile(
        self,
        sensor,  # type: ndarray
        func,  # type: str
        quant=0.99,  # type: float
        save_var_files=False,  # type: bool
    ):  # type: (...)-> Tuple[int, float]
        """Get the number of variables that explain a quantile fraction of the
        variation.

        Args:
            sensor: The sensitivity.
            func: The function name.
            quant: The quantile threshold.
            save_var_files: If True, save the influent variables indices in a NumPy file.

        Returns:
            The number of required variables
            and the threshold value for the sensitivity.
        """
        abs_vals = absolute(sensor)
        abs_sens_i = argsort(abs_vals)[::-1]
        abs_sens = abs_vals[abs_sens_i]
        total = abs_sens.sum()
        var = 0.0
        tresh_ind = 0
        while var < total * quant and tresh_ind < len(abs_sens):
            var += abs_sens[tresh_ind]
            tresh_ind += 1
        kept_vars = abs_sens_i[:tresh_ind]
        LOGGER.info("VariableInfluence for function %s", func)
        LOGGER.info(
            "Most influent variables indices to explain "
            "%% of the function variation : %s",
            int(quant * 100),
        )
        LOGGER.info(kept_vars)
        if save_var_files:
            names = self.opt_problem.design_space.variables_names
            sizes = self.opt_problem.design_space.variables_sizes
            ll_of_names = array(
                [
                    ["{}${}".format(name, i) for i in range(sizes[name])]
                    for name in names
                ]
            )
            flaten_names = array([name for sublist in ll_of_names for name in sublist])
            kept_names = flaten_names[kept_vars]
            var_names_file = "{}_influ_vars.csv".format(func)
            data = stack((kept_names, kept_vars)).T
            if PY2:
                fmt = "%s".encode("ascii")
            else:
                fmt = "%s"
            savetxt(
                var_names_file, data, fmt=fmt, delimiter=" ; ", header="name ; index"
            )
            self.output_files.append(var_names_file)
        return tresh_ind, abs_sens[tresh_ind - 1]

    def __generate_subplots(
        self,
        sens_dict,  # type: Mapping[str, ndarray]
        quantile=0.99,  # type: float
        log_scale=False,  # type: bool
        save_var_files=False,  # type: bool
    ):  # type: (...)-> Figure
        """Generate the gradients subplots from the data.

        Args:
            sens_dict: The sensors to plot.
            save_var_files: If True, save the influent variables indices in a NumPy file.

        Returns:
            The gradients subplots.

        Raises:
            ValueError: If the `sens_dict` is empty.
        """
        n_funcs = len(sens_dict)
        if n_funcs == 0:
            raise ValueError("No gradients to plot at current iteration!")

        nrows = n_funcs // 2
        if 2 * nrows < n_funcs:
            nrows += 1
        if n_funcs > 1:
            ncols = 2
        else:
            ncols = 1
        fig, axes = pyplot.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            sharey=False,
            figsize=self.DEFAULT_FIG_SIZE,
        )
        i = 0
        j = -1

        axes = atleast_2d(axes)
        n_subplots = len(axes) * len(axes[0])
        x_labels = self._generate_x_names()
        # This variable determines the number of variables to plot in the
        # x-axis. Since the data history can be edited by the user after the
        # problem was solved, we do not use something like opt_problem.dimension
        # because the problem dimension is not updated when the history is filtered.
        abscissas = range(len(tuple(sens_dict.values())[0]))

        for func, sens in sorted(sens_dict.items()):
            j += 1
            if j == ncols:
                j = 0
                i += 1
            axe = axes[i][j]
            n_vars = len(sens)
            axe.bar(abscissas, sens, color="blue", align="center")
            quant, treshold = self.__get_quantile(sens, func, quantile, save_var_files)
            axe.set_title(
                "{} variables required to explain {}% of {} variations".format(
                    quant, round(quantile * 100), func
                )
            )
            axe.set_xticklabels(x_labels, fontsize=12, rotation=90)
            axe.set_xticks(abscissas)
            axe.set_xlim(-1, n_vars + 1)
            axe.axhline(treshold, color="r")
            axe.axhline(-treshold, color="r")
            if log_scale:
                axe.set_yscale("log")

            # Update y labels spacing
            vis_labels = [
                label for label in axe.get_yticklabels() if label.get_visible() is True
            ]
            pyplot.setp(vis_labels, visible=False)
            pyplot.setp(vis_labels[::2], visible=True)

            vis_xlabels = [
                label for label in axe.get_xticklabels() if label.get_visible() is True
            ]
            if len(vis_xlabels) > 20:
                frac_xlabels = int(len(vis_xlabels) / 10.0)
                pyplot.setp(vis_xlabels, visible=False)
                pyplot.setp(vis_xlabels[::frac_xlabels], visible=True)

        if len(sens_dict) < n_subplots:
            # xlabel must be written with the same fontsize on the 2 columns
            j += 1
            axe = axes[i][j]
            axe.set_xticklabels(x_labels, fontsize=12, rotation=90)
            axe.set_xticks(abscissas)

        fig.suptitle(
            "Partial variation of the functions " + "wrt design variables", fontsize=14
        )
        return fig
