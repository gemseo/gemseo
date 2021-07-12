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
"""A scatter plot matrix to display optimization history."""
from __future__ import division, unicode_literals

import logging
from typing import Sequence

from matplotlib import pyplot
from pandas.core.frame import DataFrame

from gemseo.post.opt_post_processor import OptPostProcessor

try:
    from pandas.tools.plotting import scatter_matrix
except ImportError:
    from pandas.plotting import scatter_matrix


LOGGER = logging.getLogger(__name__)


class ScatterPlotMatrix(OptPostProcessor):
    """Scatter plot matrix among design variables, output functions and constraints.

    The list of variable names has to be passed as arguments of the plot method.
    x- and y- figure sizes can be changed in option.
    """

    def _plot(
        self,
        variables_list,  # type: Sequence[str]
        figsize_x=10,  # type: int
        figsize_y=10,  # type: int
    ):  # type: (...) -> None
        """
        Args:
            variables_list: The functions names or design variables to plot.
                If the list is empty,
                plot all design variables.
            figsize_x: The size of the figure in horizontal direction (inches).
            figsize_y: The size of the figure in vertical direction (inches).
        """

        add_dv = False
        all_funcs = self.opt_problem.get_all_functions_names()
        all_dv_names = self.opt_problem.design_space.variables_names
        variables_list.sort()

        if not variables_list:
            # In this case, plot all design variables, no functions.
            vals = self.database.get_x_history()

            # This section creates readable labels for design variables
            # i.e. toto_0, toto_1 if toto is a variable with 2 components
            x_labels = self.__get_design_var_labels(all_dv_names)

        else:
            design_variables = []
            for func in list(variables_list):
                if func not in all_funcs and func not in all_dv_names:
                    min_f = "-{}".format(func) == self.opt_problem.objective.name
                    if min_f and not self.opt_problem.minimize_objective:
                        variables_list[variables_list.index(func)] = "-{}".format(func)
                        variables_list.sort()
                    else:
                        msg = (
                            "Cannot build scatter plot matrix, "
                            "Function {} is neither among"
                            " optimization problem functions : {}"
                            " nor design variables : {}".format(
                                func, all_funcs, all_dv_names
                            )
                        )
                        raise ValueError(msg)
                if func in self.opt_problem.design_space.variables_names:
                    # if given function is a design variable, then remove it
                    add_dv = True
                    variables_list.remove(func)
                    design_variables.append(func)
            if not design_variables:
                design_variables = None

            if add_dv:
                # Sort the design variables to be consistent with GEMSEO.
                design_variables = sorted(
                    set(all_dv_names) & set(design_variables),
                    key=all_dv_names.index,
                )

                # This section creates readable labels for design variables
                # and functions i.e. toto_0, toto_1 if toto is a variable
                # with 2 components
                dv_labels = self.__get_design_var_labels(design_variables)
                if variables_list:
                    _, func_labels, _ = self.database.get_history_array(
                        functions=variables_list,
                        design_variables_names=None,
                        add_dv=False,
                    )
                else:
                    func_labels = []
                # vname contains function names + condensed variable names
                # i.e. "toto" even if toto has 2 components or more
                vname = variables_list + design_variables
                # x_labels contains function names + readable variable names
                x_labels = func_labels + dv_labels
            else:
                # In this case we are only plotting functions.
                # Functions have unique names, so x_labels and
                # vname are equal.
                vname = variables_list
                _, x_labels, _ = self.database.get_history_array(
                    functions=variables_list,
                    design_variables_names=None,
                    add_dv=False,
                )
                x_labels.sort()
            dataset = self.opt_problem.export_to_dataset("OptimizationProblem")
            vals = dataset.get_data_by_names(vname, False)
        # Next line is a trick for a bug workaround in numpy/matplotlib
        # https://stackoverflow.com/questions/39180873/
        # pandas-dataframe-valueerror-num-must-be-1-num-0-not-1
        vals = (list(x) for x in vals)
        frame = DataFrame(vals, columns=x_labels)
        scatter_matrix(frame, alpha=1.0, figsize=(figsize_x, figsize_y), diagonal="kde")
        fig = pyplot.gcf()
        fig.tight_layout()
        self._add_figure(fig)

    def __get_design_var_labels(self, des_vars):
        """Create labels for design variables.

        Args:
            des_vars (list(str)): The design variables to get its labels.

        Returns:
            list(str): The labels for the design variables.
        """

        dv_names = []
        for d_v in des_vars:
            dv_size = self.opt_problem.design_space.variables_sizes[d_v]
            if dv_size == 1:
                dv_names.append(d_v)
            else:
                for k in range(dv_size):
                    dv_names.append("{}_{}".format(d_v, k))
        return dv_names
