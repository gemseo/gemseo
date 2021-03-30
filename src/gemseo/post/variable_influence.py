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
"""
Plot the partial sensitivity of the functions
*********************************************
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from matplotlib import pyplot
from numpy import absolute, arange, argsort, array, atleast_2d, savetxt, stack

from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.py23_compat import PY2

standard_library.install_aliases()


from gemseo import LOGGER


class VariableInfluence(OptPostProcessor):
    """
    The **VariableInfluence** post processing
    performs first order variable influence analysis
    by computing df/dxi * (xi* - xi0)
    where xi0 is the initial value of the variable
    and xi* is the optimal value of the variable

    Options of the plot method are the x- and y- figure sizes,
    the quantile level, the use of a logarithmic scale and
    the possibility to save the influent variables indices
    as a numpy file
    It is also possible either to save the plot, to show the plot or both.
    """

    def _plot(
        self,
        figsize_x=20,
        figsize_y=5,
        quantile=0.99,
        absolute_value=False,
        log_scale=False,
        save_var_files=False,
        show=False,
        save=False,
        file_path="var_infl",
        extension="pdf",
    ):
        """
        Plots the ScatterPlotMatrix graph

        :param figsize_x: size of figure in horizontal direction (inches)
        :type figsize_x: int
        :param figsize_y: size of figure in vertical direction (inches)
        :type figsize_y: int
        :param quantile: between 0 and  1, proportion of the total
            sensitivity to use as a threshold to filter the variables
        :type quantile: float
        :param log_scale: if True, use a logarithmic scale
        :type log_scale: bool
        :param absolute_value: if true, plot the absolute value of the
            influence
        :type absolute_value: bool
        :param save_var_files: save the influent variables indices as a numpy
            file
        :type save_var_files: bool
        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param extension: file extension
        :type extension: str
        """
        all_funcs = self.opt_problem.get_all_functions_names()
        _, x_opt, _, _, _ = self.opt_problem.get_optimum()
        x_0 = self.database.get_x_by_iter(0)
        if log_scale:
            absolute_value = True
        sens_dict = {}
        for func in all_funcs:
            grad = self.database.get_f_of_x(self.database.GRAD_TAG + func, x_0)
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
                        sens_dict[func + "_" + str(i)] = sens

        fig = self.__generate_subplots(
            sens_dict, figsize_x, figsize_y, quantile, log_scale, save_var_files
        )
        self._save_and_show(
            fig, save=save, show=show, file_path=file_path, extension=extension
        )

    def __get_quantile(self, sensor, func, quant=0.99, save_var_files=False):
        """
        Computes the number of variables to keep that explain quant fraction
        of the variation

        :param sensor: the numpy array containing the sensitivity
        :param func: the function name
        :param quant: the quantile treshold
        :param save_var_files: save the influent variables indices as a numpy
            file
        :returns: the number of required variables and the treshold value
            for the sensitivity
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
                [[name + "$" + str(i) for i in range(sizes[name])] for name in names]
            )
            flaten_names = array([name for sublist in ll_of_names for name in sublist])
            kept_names = flaten_names[kept_vars]
            var_names_file = func + "_influ_vars.csv"
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
        sens_dict,
        figsize_x,
        figsize_y,
        quantile=0.99,
        log_scale=False,
        save_var_files=False,
    ):
        """
        Generates the gradient sub plots from the data

        :param x_ref: reference value for x
        :param sens_dict: dict of sensors to plot
        :param figsize_x: size of figure in horizontal direction (inches)
        :param figsize_y: size of figure in vertical direction (inches)
        :param save_var_files: save the influent variables indices as a numpy
            file
        """
        n_funcs = len(sens_dict)
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
            figsize=(figsize_x, figsize_y),
        )
        i = 0
        j = -1

        axes = atleast_2d(axes)
        n_subplots = len(axes) * len(axes[0])

        for func, sens in sorted(sens_dict.items()):
            j += 1
            if j == ncols:
                j = 0
                i += 1
            axe = axes[i][j]
            n_vars = len(sens)
            abscissa = arange(n_vars)
            # x_labels = [r'$x_{' + str(x_id) + '}$' for x_id in abscissa]
            x_labels = [str(x_id) for x_id in abscissa]
            axe.bar(abscissa, sens, color="blue", align="center")
            quant, treshold = self.__get_quantile(sens, func, quantile, save_var_files)
            axe.set_title(
                str(quant)
                + " variables required"
                + " to explain "
                + str(round(quantile * 100))
                + "% of "
                + func
                + " variations"
            )
            axe.set_xticklabels(x_labels, fontsize=14)
            axe.set_xticks(abscissa)
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
            axe.set_xticklabels(x_labels, fontsize=14)
            axe.set_xticks(abscissa)

        fig.suptitle(
            "Partial variation of the functions " + "wrt design variables", fontsize=14
        )
        return fig
