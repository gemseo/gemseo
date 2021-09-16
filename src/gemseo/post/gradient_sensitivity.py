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
"""Plot the derivatives of the functions."""
from __future__ import division, unicode_literals

import logging
from typing import Dict, Iterable, Mapping

from matplotlib import pyplot
from matplotlib.figure import Figure
from numpy import arange, atleast_2d, ndarray

from gemseo.post.opt_post_processor import OptPostProcessor

LOGGER = logging.getLogger(__name__)


class GradientSensitivity(OptPostProcessor):
    """Histograms of the derivatives of objective and constraints.

    The plot method considers the derivatives at the last iteration. The iteration can
    be changed.
    """

    DEFAULT_FIG_SIZE = (10.0, 10.0)

    def _plot(
        self,
        iteration=-1,  # type: int
        scale_gradients=False,  # type: bool
    ):  # type: (...) -> None
        """
        Args:
            iteration: The iteration to plot the sensitivities;
                if negative, use the optimum.
            scale_gradients: If True, normalize each gradient
                w.r.t. the design variables.
        """
        if iteration == -1:
            x_ref = self.opt_problem.solution.x_opt
        else:
            x_ref = self.opt_problem.database.get_x_by_iter(iteration)
        grad_dict = self.__get_grad_dict(x_ref, scale_gradients=scale_gradients)

        x_names = self._generate_x_names()

        fig = self.__generate_subplots(
            x_names,
            x_ref,
            grad_dict,
            scale_gradients=scale_gradients,
        )
        self._add_figure(fig)

    def __get_grad_dict(
        self,
        x_ref,  # type: ndarray
        scale_gradients=False,  # type: bool
    ):  # type: (...) -> Dict[str,ndarray]
        """Create a gradient dictionary from a given iteration.

        Scale it if necessary.

        Args:
            x_ref: The reference value for x.
            scale_gradients: If True, normalize each gradient
                w.r.t. the design variables.

        Returns:
            The gradients of the outputs
            indexed by the names of the output,
            e.g. 'output_name' for a mono-dimensional output,
            or 'output_name_i' for the i-th component of a multi-dimensional output.
        """
        all_funcs = self.opt_problem.get_all_functions_names()
        scale_func = self.opt_problem.design_space.unnormalize_vect
        grad_dict = {}
        for func in all_funcs:
            grad = self.database.get_f_of_x("@{}".format(func), x_ref)
            if grad is not None:
                if len(grad.shape) == 1:
                    if scale_gradients:
                        grad = scale_func(grad, minus_lb=False)
                    grad_dict[func] = grad
                else:
                    n_f, _ = grad.shape
                    for i in range(n_f):
                        if scale_gradients:
                            grad[i, :] = scale_func(grad[i, :], minus_lb=False)
                        grad_dict["{}_{}".format(func, i)] = grad[i, :]
        return grad_dict

    @classmethod
    def __generate_subplots(
        cls,
        x_names,  # type: Iterable[str]
        x_ref,  # type: ndarray
        grad_dict,  # type: Mapping[str, ndarray]
        scale_gradients=False,  # type: bool
    ):  # type: (...)-> Figure
        """Generate the gradients subplots from the data.

        Args:
            x_names: The variables names.
            x_ref: The reference value for x.
            grad_dict: The gradients to plot.
            scale_gradients: If True, normalize the gradients w.r.t. the design variables.

        Returns:
            The gradients subplots.

        Raises:
            ValueError: If `grad_dict` is empty.
        """
        n_funcs = len(grad_dict)
        if n_funcs == 0:
            raise ValueError("No gradients to plot at current iteration!")
        nrows = n_funcs // 2
        if 2 * nrows < n_funcs:
            nrows += 1
        ncols = 2
        fig, axes = pyplot.subplots(
            nrows=nrows,
            ncols=2,
            sharex=True,
            sharey=False,
            figsize=cls.DEFAULT_FIG_SIZE,
        )
        i = 0
        j = -1

        axes = atleast_2d(axes)
        n_subplots = len(axes) * len(axes[0])
        abscissa = arange(len(x_ref))
        x_labels = [str(x_id) for x_id in x_names]
        for func, grad in sorted(grad_dict.items()):
            j += 1
            if j == ncols:
                j = 0
                i += 1
            axe = axes[i][j]
            axe.bar(abscissa, grad, color="blue", align="center")
            axe.set_title(func)
            axe.set_xticklabels(x_labels, fontsize=12, rotation=90)
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
            axe.set_xticklabels(x_labels, fontsize=12, rotation=90)
            axe.set_xticks(abscissa)

        if scale_gradients:
            fig.suptitle(
                "Derivatives of objective and constraints"
                + " with respect to design variables.\n \nNormalized Design Space.",
                fontsize=14,
            )
        else:
            fig.suptitle(
                "Derivatives of objective and constraints"
                + " with respect to design variables",
                fontsize=14,
            )
        return fig
