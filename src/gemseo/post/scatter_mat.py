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
from __future__ import annotations

import logging
from typing import Sequence

from matplotlib import pyplot
from numpy import any
from pandas.core.frame import DataFrame
from pandas.plotting import scatter_matrix

from gemseo.post.opt_post_processor import OptPostProcessor


LOGGER = logging.getLogger(__name__)


class ScatterPlotMatrix(OptPostProcessor):
    """Scatter plot matrix among design variables, output functions and constraints.

    The list of variable names has to be passed as arguments of the plot method.
    """

    DEFAULT_FIG_SIZE = (10.0, 10.0)

    def _plot(
        self,
        variable_names: Sequence[str],
        filter_non_feasible: bool = False,
    ) -> None:
        """
        Args:
            variable_names: The functions names or design variables to plot.
                If the list is empty,
                plot all design variables.
            filter_non_feasible: If True, remove the non-feasible
                points from the data.

        Raises:
            ValueError: If `filter_non_feasible` is set to True and no feasible
                points exist. If an element from variable_names is not either
                a function or a design variable.
        """  # noqa: D205, D212, D415
        problem = self.opt_problem
        add_design_variables = False
        all_function_names = problem.get_all_functions_names()
        all_design_names = problem.design_space.variables_names

        if not problem.minimize_objective and self._obj_name in variable_names:
            obj_index = variable_names.index(self._obj_name)
            variable_names[obj_index] = self._standardized_obj_name

        variable_names.sort()
        if not variable_names:
            # In this case, plot all design variables, no functions.
            variable_values = problem.get_data_by_names(
                names=all_design_names,
                as_dict=False,
                filter_non_feasible=filter_non_feasible,
            )
            variable_labels = self._generate_x_names(variables=all_design_names)

        else:
            design_names = []
            function_names = []
            for variable_name in list(variable_names):
                if (
                    variable_name not in all_function_names
                    and variable_name not in all_design_names
                    and variable_name not in problem.constraint_names
                ):
                    raise ValueError(
                        "Cannot build scatter plot matrix: "
                        f"function {variable_name} is neither among "
                        f"optimization problem functions: {all_function_names} "
                        f"nor design variables: {all_design_names}"
                    )

                if variable_name in problem.design_space.variables_names:
                    add_design_variables = True
                    design_names.append(variable_name)
                elif variable_name in problem.constraint_names:
                    function_names.extend(problem.constraint_names[variable_name])
                else:
                    function_names.append(variable_name)

            if not design_names:
                design_names = None

            if add_design_variables:
                # Sort the design variables to be consistent with GEMSEO.
                design_names = sorted(
                    set(all_design_names) & set(design_names),
                    key=all_design_names.index,
                )

                design_labels = self._generate_x_names(variables=design_names)
                if function_names:
                    _, function_labels, _ = self.database.get_history_array(
                        functions=function_names, add_dv=False
                    )
                else:
                    function_labels = []

                variable_names = function_names + design_names
                variable_labels = function_labels + design_labels
            else:
                variable_names = function_names
                _, variable_labels, _ = self.database.get_history_array(
                    functions=variable_names, add_dv=False
                )
                variable_labels.sort()

            variable_values = problem.get_data_by_names(
                names=variable_names,
                as_dict=False,
                filter_non_feasible=filter_non_feasible,
            )

        if (
            self._standardized_obj_name in variable_labels
            and not problem.minimize_objective
            and not problem.use_standardized_objective
        ):
            index = variable_labels.index(self._standardized_obj_name)
            variable_labels[index] = self._obj_name
            variable_values[:, index] *= -1

        if filter_non_feasible and not any(variable_values):
            raise ValueError("No feasible points were found.")

        # Next line is a trick for a bug workaround in numpy/matplotlib
        # https://stackoverflow.com/questions/39180873/
        # pandas-dataframe-valueerror-num-must-be-1-num-0-not-1
        scatter_matrix(
            DataFrame((list(x.real) for x in variable_values), columns=variable_labels),
            alpha=1.0,
            figsize=self.DEFAULT_FIG_SIZE,
            diagonal="kde",
        )
        fig = pyplot.gcf()
        fig.tight_layout()
        self._add_figure(fig)
