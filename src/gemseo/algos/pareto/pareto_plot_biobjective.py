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
"""Plotting a 2D Pareto front."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import ndarray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from gemseo.typing import BooleanArray


class ParetoPlotBiObjective:
    """Plot a 2D Pareto front on a Matplotlib axes."""

    def __init__(
        self,
        ax: Axes,
        obj_values: ndarray,
        pareto_optimal_loc: BooleanArray,
        obj_names: Sequence[str],
        all_pareto_optimal: BooleanArray,
        is_non_feasible: BooleanArray,
        bi_obj: bool = False,
        show_non_feasible: bool = True,
    ) -> None:
        """
        Args:
            ax: The :class:`~matplotlib.axes.Axes` on which to plot.
            obj_values: The objective function array, of size (n_samples, n_objs).
            pareto_optimal_loc: A vector of booleans of size n_samples,
                True if the point is Pareto optimal.
            obj_names: The names of the objectives.
            all_pareto_optimal: The indices of points that are Pareto optimal
                w.r.t. all criteria.
            is_non_feasible: An Array of booleans of size n_samples,
                True if non_feasible.
            bi_obj: True if there are only two objective values.
            show_non_feasible: True to show the non-feasible points.
        """  # noqa: D205, D212, D415
        self.__nb_pareto_pts = obj_values.shape[0]
        self.__ax = ax
        self.__obj_values = obj_values
        self.__pareto_optimal_loc = pareto_optimal_loc
        self.__obj_names = obj_names
        self.__pareto_optimal_all = all_pareto_optimal
        self.__is_non_feasible = is_non_feasible
        self.__bi_objective = bi_obj
        self.__show_non_feasible = show_non_feasible
        self.__pareto_dominated_indexes = None

        # Compute the Pareto dominated indexes
        self.__compute_pareto_dominated_indexes()

    def plot_on_axes(self) -> None:
        """Plot the Pareto points on the Matplolib axes."""
        self.__plot_pareto_dominated_points()
        self.__plot_non_feasible_points()
        self.__plot_globally_pareto_optimal_points()
        self.__plot_locally_pareto_optimal_points()
        self.__ax.set_xlabel(self.__obj_names[0])
        self.__ax.set_ylabel(self.__obj_names[1])

    def __compute_pareto_dominated_indexes(self) -> None:
        """Compute the Pareto dominated indexes.

        The Pareto-dominated points are all the points which are feasible, but not
        locally and globally pareto optimal.
        """
        pareto_dominated_indexes = array([True] * self.__nb_pareto_pts)
        self.__pareto_dominated_indexes = (
            pareto_dominated_indexes
            & ~array(self.__is_non_feasible)
            & ~self.__pareto_optimal_loc
            & ~self.__pareto_optimal_all
        )

    def __plot_pareto_dominated_points(self) -> None:
        """Plot the Pareto-dominated points on the scatter plot."""
        if True in self.__pareto_dominated_indexes:
            self.__ax.scatter(
                self.__obj_values[self.__pareto_dominated_indexes, 0],
                self.__obj_values[self.__pareto_dominated_indexes, 1],
                color="b",
                label="Pareto dominated",
            )

    def __plot_non_feasible_points(self) -> None:
        """Plot the non-feasible points on the scatter plot."""
        if True in self.__is_non_feasible and self.__show_non_feasible:
            self.__ax.scatter(
                self.__obj_values[self.__is_non_feasible, 0],
                self.__obj_values[self.__is_non_feasible, 1],
                color="r",
                marker="X",
                label="Non feasible point",
            )

    def __plot_globally_pareto_optimal_points(self) -> None:
        """Plot the globally optimal Pareto points on the scatter plot."""
        if True in self.__pareto_optimal_all:
            if self.__bi_objective:
                label = "Pareto optimal"
            else:
                label = "Globally Pareto optimal"
            self.__ax.scatter(
                self.__obj_values[self.__pareto_optimal_all, 0],
                self.__obj_values[self.__pareto_optimal_all, 1],
                color="g",
                label=label,
            )

    def __plot_locally_pareto_optimal_points(self) -> None:
        """Plot the locally optimal Pareto points on the scatter plot."""
        if True in self.__pareto_optimal_loc and not self.__bi_objective:
            self.__ax.scatter(
                self.__obj_values[self.__pareto_optimal_loc, 0],
                self.__obj_values[self.__pareto_optimal_loc, 1],
                color="r",
                label="Locally Pareto optimal",
            )
