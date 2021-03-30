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
Plot the derivatives of the functions
*************************************
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from matplotlib import pyplot
from numpy import arange, atleast_2d

from gemseo.post.opt_post_processor import OptPostProcessor

standard_library.install_aliases()


from gemseo import LOGGER


class GradientSensitivity(OptPostProcessor):
    """
    The **GradientSensitivity** post processing
    builds histograms of derivatives of objective and constraints

    The plot method considers the derivatives at the last iteration.
    The iteration can be changed in option. The x- and y- figure sizes
    can also be modified in option.
    It is possible either to save the plot, to show the plot or both.
    """

    def _plot(
        self,
        iteration=-1,
        figsize_x=10,
        figsize_y=10,
        save=False,
        show=False,
        file_path="gradient_sensitivity",
        extension="pdf",
    ):
        """
        Plots the GradientSensitivity graph

        :param iteration:  the iteration to plot sensitivities, if negative,
            use optimum
        :type iteration: int
        :param figsize_x: size of figure in horizontal direction (inches)
        :type figsize_x: int
        :param figsize_y: size of figure in vertical direction (inches)
        :type figsize_y: int
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

        if iteration == -1:
            x_ref = self.opt_problem.solution.x_opt
        else:
            x_ref = self.opt_problem.database.get_x_by_iter(iteration)
        grad_dict = {}
        for func in all_funcs:
            grad = self.database.get_f_of_x("@" + func, x_ref)
            if grad is not None:
                if len(grad.shape) == 1:
                    grad_dict[func] = grad
                else:
                    n_f, _ = grad.shape
                    for i in range(n_f):
                        grad_dict[func + "_" + str(i)] = grad[i, :]

        fig = self.__generate_subplots(x_ref, grad_dict, figsize_x, figsize_y)
        self._save_and_show(
            fig, save=save, show=show, file_path=file_path, extension=extension
        )

    @staticmethod
    def __generate_subplots(x_ref, grad_dict, figsize_x=10, figsize_y=10):
        """
        Generates the gradient sub plots from the data

        :param x_ref: reference value for x
        :param grad_dict : dict of gradients to plot
        :param figsize_x : size of figure in horizontal direction (inches)
        :param figsize_y : size of figure in vertical direction (inches)
        """
        n_funcs = len(grad_dict)
        if n_funcs == 0:
            raise ValueError("No gradients to plot at current iteration !")
        nrows = n_funcs // 2
        if 2 * nrows < n_funcs:
            nrows += 1
        ncols = 2
        fig, axes = pyplot.subplots(
            nrows=nrows,
            ncols=2,
            sharex=True,
            sharey=False,
            figsize=(figsize_x, figsize_y),
        )
        i = 0
        j = -1

        axes = atleast_2d(axes)
        n_subplots = len(axes) * len(axes[0])
        abscissa = arange(len(x_ref))
        x_labels = [r"$x_{" + str(x_id) + "}$" for x_id in abscissa]
        for func, grad in sorted(grad_dict.items()):
            j += 1
            if j == ncols:
                j = 0
                i += 1
            axe = axes[i][j]
            axe.bar(abscissa, grad, color="blue", align="center")
            axe.set_title(func)
            axe.set_xticklabels(x_labels, fontsize=14)
            axe.set_xticks(abscissa)
            # Update y labels spacing
            vis_labels = [
                label for label in axe.get_yticklabels() if label.get_visible() is True
            ]
            pyplot.setp(vis_labels[::2], visible=False)

        if len(grad_dict) < n_subplots:
            # xlabel must be written with the same fontsize on the 2 columns
            j += 1
            #             if j == ncols: Seems impossible to reach
            #                 j = 0
            #                 i += 1
            axe = axes[i][j]
            axe.set_xticklabels(x_labels, fontsize=14)
            axe.set_xticks(abscissa)

        fig.suptitle(
            "Derivatives of objective and constraints"
            + " with respect to design variables",
            fontsize=14,
        )
        return fig
