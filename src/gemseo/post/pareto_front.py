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
"""A Pareto Front."""
from __future__ import annotations

import logging
from typing import Sequence

from numpy import full
from numpy import ndarray

from gemseo.algos.pareto_front import generate_pareto_plots
from gemseo.post.opt_post_processor import OptPostProcessor

LOGGER = logging.getLogger(__name__)


class ParetoFront(OptPostProcessor):
    """Compute the Pareto front for a multi-objective problem.

    The Pareto front of an optimization problem is the set of ``non-dominated`` points of
    the design space for which there is no other point that improves an objective
    without damaging another.

    This post-processing computes the Pareto front and generates a matrix of plots,
    one per couple of objectives.
    For a given plot, the red markers are the non-dominated points according to the
    objectives of this plot and the green markers are the non-dominated points
    according to all the objectives.
    The latter are also called ``Pareto optimal points``.
    """

    DEFAULT_FIG_SIZE = (10.0, 10.0)

    def _plot(
        self,
        objectives: Sequence[str] | None = None,
        objectives_labels: Sequence[str] | None = None,
        show_non_feasible: bool = True,
    ) -> None:
        """
        Args:
            objectives: The functions names or design variables to plot.
                If None, use the objective function (maybe a vector).
            objectives_labels: The labels of the objective components.
                If None, use the objective name suffixed by an index.
            show_non_feasible: If True, show the non-feasible points in the plot.

        Raises:
            ValueError: If the numbers of objectives and objectives
                labels are different.
        """  # noqa: D205, D212, D415
        if objectives is None:
            objectives = [self.opt_problem.objective.name]

        all_funcs = self.opt_problem.get_all_functions_names()
        all_dv_names = self.opt_problem.design_space.variables_names

        sample_values, all_labels = self.__compute_names_and_values(
            all_dv_names, all_funcs, objectives
        )

        non_feasible_samples = self.__compute_non_feasible_samples(sample_values)

        if objectives_labels is not None:
            if len(all_labels) != len(objectives_labels):
                raise ValueError(
                    "objective_labels shall have the same dimension as the number"
                    " of objectives to plot."
                )
            all_labels = objectives_labels

        fig = generate_pareto_plots(
            sample_values,
            all_labels,
            fig_size=self.DEFAULT_FIG_SIZE,
            non_feasible_samples=non_feasible_samples,
            show_non_feasible=show_non_feasible,
        )

        self._add_figure(fig)

    def __compute_names_and_values(
        self,
        all_dv_names: Sequence[str],
        all_funcs: Sequence[str],
        objectives: Sequence[str],
    ) -> tuple[ndarray, list[str]]:
        """Compute the names and values of the objective and design variables.

        Args:
             all_dv_names: The design variables names.
             all_funcs: The function names.
             objectives: The objective names.

        Returns:
            The sample values and the sample names.
        """
        design_variables = []
        for func in list(objectives):
            self.__check_objective_name(all_dv_names, all_funcs, func, objectives)
            self.__move_objective_to_design_variable(design_variables, func, objectives)

        if not design_variables:
            design_variables_labels = []
            all_data_names = objectives
            _, objective_labels, _ = self.database.get_history_array(
                functions=objectives, add_dv=False
            )
        elif not objectives:
            design_variables_labels = self._generate_x_names(variables=design_variables)
            all_data_names = design_variables
            objective_labels = []
        else:
            design_variables_labels = self._generate_x_names(variables=design_variables)
            all_data_names = objectives + design_variables
            _, objective_labels, _ = self.database.get_history_array(
                functions=objectives, add_dv=False
            )

        all_data_names.sort()
        all_labels = sorted(objective_labels + design_variables_labels)

        sample_values = self.opt_problem.get_data_by_names(
            names=all_data_names, as_dict=False
        )

        return sample_values, all_labels

    def __check_objective_name(
        self,
        all_dv_names: Sequence[str],
        all_funcs: Sequence[str],
        func: str,
        objectives: Sequence[str],
    ) -> None:
        """Check that the objective name is valid.

        Args:
             all_dv_names: The design variables names.
             all_funcs: The function names.
             func: The function name.
             objectives: The objectives names.

        Raises:
            ValueError: If the objective name is not valid.
        """
        if func not in all_funcs and func not in all_dv_names:
            min_f = "-" + func == self.opt_problem.objective.name
            if min_f and not self.opt_problem.minimize_objective:
                objectives[objectives.index(func)] = "-" + func
            else:
                msg = (
                    "Cannot build Pareto front,"
                    " Function {} is neither among"
                    " optimization problem functions: "
                    "{} nor design variables: {}."
                )
                msg = msg.format(func, str(all_funcs), str(all_dv_names))
                raise ValueError(msg)

    def __move_objective_to_design_variable(
        self,
        design_variables: Sequence[str],
        func: str,
        objectives: Sequence[str],
    ) -> None:
        """Move an objective to a design variable.

        If the given function is a design variable,
        then move it from the objectives to the design_variables.

        Args:
             design_variables: The design variables.
             func: The function name.
             objectives: The objectives names.
        """
        if func in self.opt_problem.design_space.variables_names:
            objectives.remove(func)
            design_variables.append(func)

    def __compute_non_feasible_samples(self, sample_values: ndarray) -> ndarray:
        """Compute the non-feasible indexes.

        Args:
            sample_values: The sample values.

        Returns:
            An array of size ``n_samples``, True if the point is non-feasible.
        """
        x_feasible, _ = self.opt_problem.get_feasible_points()
        feasible_indexes = [self.database.get_index_of(x) for x in x_feasible]

        is_non_feasible = full(sample_values.shape[0], True)
        is_non_feasible[feasible_indexes] = False

        return is_non_feasible
