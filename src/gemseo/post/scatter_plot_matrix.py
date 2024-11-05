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

from typing import ClassVar

from matplotlib import pyplot
from numpy import any as np_any
from pandas.core.frame import DataFrame
from pandas.plotting import scatter_matrix

from gemseo.post.base_post import BasePost
from gemseo.post.scatter_plot_matrix_settings import ScatterPlotMatrix_Settings


class ScatterPlotMatrix(BasePost[ScatterPlotMatrix_Settings]):
    """Scatter plot matrix among design variables, output functions and constraints.

    The list of variable names has to be passed as arguments of the plot method.
    """

    Settings: ClassVar[type[ScatterPlotMatrix_Settings]] = ScatterPlotMatrix_Settings

    def _plot(self, settings: ScatterPlotMatrix_Settings) -> None:
        """
        Raises:
            ValueError: If `filter_non_feasible` is set to True and no feasible
                points exist. If an element from variable_names is not either
                a function or a design variable.
        """  # noqa: D205, D212, D415
        variable_names = list(settings.variable_names)
        filter_non_feasible = settings.filter_non_feasible

        problem = self.optimization_problem
        add_design_variables = False
        all_function_names = problem.function_names
        all_design_names = problem.design_space.variable_names

        if not problem.minimize_objective and self._obj_name in variable_names:
            obj_index = variable_names.index(self._obj_name)
            variable_names[obj_index] = self._standardized_obj_name

        variable_names.sort()
        if not variable_names:
            # In this case, plot all design variables, no functions.
            variable_values = problem.history.get_data_by_names(
                names=all_design_names,
                as_dict=False,
                filter_non_feasible=filter_non_feasible,
            )
            variable_labels = self._get_design_variable_names(
                variables=all_design_names
            )

        else:
            design_names = []
            function_names = []
            for variable_name in list(variable_names):
                if (
                    variable_name not in all_function_names
                    and variable_name not in all_design_names
                    and variable_name
                    not in problem.constraints.original_to_current_names
                ):
                    msg = (
                        "Cannot build scatter plot matrix: "
                        f"function {variable_name} is neither among "
                        f"optimization problem functions: {all_function_names} "
                        f"nor design variables: {all_design_names}"
                    )
                    raise ValueError(msg)

                if variable_name in problem.design_space:
                    add_design_variables = True
                    design_names.append(variable_name)
                elif variable_name in problem.constraints.original_to_current_names:
                    function_names.extend(
                        problem.constraints.original_to_current_names[variable_name]
                    )
                else:
                    function_names.append(variable_name)

            if add_design_variables:
                # Sort the design variables to be consistent with GEMSEO.
                design_names = sorted(
                    set(all_design_names) & set(design_names),
                    key=all_design_names.index,
                )

                design_labels = self._get_design_variable_names(variables=design_names)
                if function_names:
                    _, function_labels, _ = self.database.get_history_array(
                        function_names=function_names, with_x_vect=False
                    )
                else:
                    function_labels = []

                variable_names = function_names + design_names
                variable_labels = function_labels + design_labels
            else:
                variable_names = function_names
                _, variable_labels, _ = self.database.get_history_array(
                    function_names=variable_names, with_x_vect=False
                )
                variable_labels.sort()

            variable_values = problem.history.get_data_by_names(
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

        if filter_non_feasible and not np_any(variable_values):
            msg = "No feasible points were found."
            raise ValueError(msg)

        # Next line is a trick for a bug workaround in numpy/matplotlib
        # https://stackoverflow.com/questions/39180873/
        # pandas-dataframe-valueerror-num-must-be-1-num-0-not-1
        scatter_matrix(
            DataFrame((list(x.real) for x in variable_values), columns=variable_labels),
            alpha=1.0,
            figsize=settings.fig_size,
            diagonal="kde",
        )
        fig = pyplot.gcf()
        fig.tight_layout()
        self._add_figure(fig)
