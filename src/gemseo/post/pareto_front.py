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
"""A Pareto Front."""
from __future__ import division, unicode_literals

import logging
from typing import List, Optional, Sequence, Tuple

from numpy import full, ndarray

from gemseo.algos.pareto_front import generate_pareto_plots
from gemseo.post.opt_post_processor import OptPostProcessor

LOGGER = logging.getLogger(__name__)


class ParetoFront(OptPostProcessor):
    """Compute the Pareto front Search for all non dominated points.

    For each point, check if it exists ``j`` such that there is no lower value
    for ``obj_values[:,j]`` that does not degrade
    at least one other objective ``obj_values[:,i]``.

    Generate a plot or a matrix of plots if there are more than 2 objectives.
    Plot in red the locally non dominated points for the currrent two objectives.
    Plot in green the globally (all objectives) Pareto optimal points.
    """

    def _plot(
        self,
        objectives=None,  # type: Optional[Sequence[str]]
        objectives_labels=None,  # type: Optional[Sequence[str]]
        figsize_x=10.0,  # type: float
        figsize_y=10.0,  # type: float
        show_non_feasible=True,  # type: bool
    ):  # type: (...) -> None
        """
        Args:
            objectives: The functions names or design variables to plot.
                If None, use the objective function (may be a vector).
            objectives_labels: The labels of the objective components.
                If None, use the objective name suffixed by an index.
            figsize_x: The size of figure in the horizontal direction (inches).
            figsize_y: The size of figure in the vertical direction (inches).
            show_non_feasible: If True, show the non feasible points in the plot.

        Raises:
            ValueError: If the numbers of objectives and objectives
             labels are different.
        """
        if objectives is None:
            objectives = [self.opt_problem.objective.name]

        all_funcs = self.opt_problem.get_all_functions_names()
        all_dv_names = self.opt_problem.design_space.variables_names

        vals, vname = self.__compute_names_and_values(
            all_dv_names, all_funcs, objectives
        )

        non_feasible_samples = self.__compute_non_feasible_samples(vals)

        if objectives_labels is not None:
            if len(vname) != len(objectives_labels):
                raise ValueError(
                    "objective_labels shall have the same dimension as vname."
                )
            vname = objectives_labels

        fig = generate_pareto_plots(
            vals,
            vname,
            figsize=(figsize_x, figsize_y),
            non_feasible_samples=non_feasible_samples,
            show_non_feasible=show_non_feasible,
        )

        self._add_figure(fig)

    def __compute_names_and_values(
        self,
        all_dv_names,  # type: Sequence[str]
        all_funcs,  # type: Sequence[str]
        objectives,  # type: Sequence[str]
    ):  # type: (...) -> Tuple[ndarray,List[str]]
        """Compute the names and values of the objective and design variables.

        Args:
             add_dv_names: The design variables names.
             all_funcs: The function names.
             objectives: The objective names.

        Returns:
            The sample values and the sample names.
        """
        # TODO: Those lines are not covered by the tests. It has to be
        # investigated to see if it's dead code.
        if not objectives:
            # The function list only contains design variables
            vals = self.database.get_x_history()
            vname = self.database.set_dv_names(vals[0].shape[0])
        else:
            design_variables = []
            for func in list(objectives):
                self.__check_objective_name(all_dv_names, all_funcs, func, objectives)
                self.__move_objective_to_design_variable(
                    design_variables, func, objectives
                )

            if not design_variables:
                design_variables = None
                add_dv = False
            else:
                add_dv = True

            vals, vname, _ = self.database.get_history_array(
                objectives, design_variables, add_dv=add_dv
            )
        return vals, vname

    def __check_objective_name(
        self,
        all_dv_names,  # type: Sequence[str]
        all_funcs,  # type: Sequence[str]
        func,  # type: str
        objectives,  # type: Sequence[str]
    ):  # type: (...) -> None
        """Check that the objective name is valid.

        Args:
             add_dv_names: The design variables names.
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
        design_variables,  # type: Sequence[str]
        func,  # type: str
        objectives,  # type: Sequence[str]
    ):  # type: (...) -> None
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

    def __compute_non_feasible_samples(
        self, vals  # type: ndarray
    ):  # type: (...) -> ndarray
        """Compute the non-feasible indexes.

        Args:
            vals: The sample values.

        Returns:
              An array of size ``n_samples``, True if the point is non feasible
        """
        x_feasible, _ = self.opt_problem.get_feasible_points()
        feasible_indexes = [self.database.get_index_of(x) for x in x_feasible]

        is_non_feasible = full(vals.shape[0], True)
        is_non_feasible[feasible_indexes] = False

        return is_non_feasible
