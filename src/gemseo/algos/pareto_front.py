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
"""Compute and display a Pareto Front."""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
from numpy import all as np_all
from numpy import any as np_any
from numpy import array
from numpy import full
from numpy import ndarray
from numpy import vstack


def compute_pareto_optimal_points(
    obj_values: ndarray,
    feasible_points: ndarray | None = None,
) -> ndarray:
    """Compute the Pareto optimal points.

    Search for all the non-dominated points, i.e. there exists ``j`` such that
    there is no lower value for ``obj_values[:,j]`` that does not degrade
    at least one other objective ``obj_values[:,i]``.

    Args:
        obj_values: The objective function array of size `(n_samples, n_objs)`.
        feasible_points: An array of boolean of size n_sample,
            True if the sample is feasible,
            False otherwise.

    Returns:
        The vector of booleans of size n_samples, True if
        the point is Pareto optimal.
    """
    pareto_optimal = full(obj_values.shape[0], True)

    if feasible_points is None:
        feasible_points = full(obj_values.shape[0], True)

    def any_ax1_all(arr):
        return np_all(np_any(arr, axis=1))

    # Store the feasible indexes
    feasible_indexes = []
    for i, feasible_point in enumerate(feasible_points):
        if not feasible_point:
            pareto_optimal[i] = False
        else:
            feasible_indexes.append(i)

    # Exclude the non-feasible points for the computation of the Pareto optimal points
    obj_values_filtered = obj_values[feasible_indexes, :]
    for i, feasible_index in enumerate(feasible_indexes):
        obj = obj_values[feasible_index]
        before_are_worse = any_ax1_all(obj_values_filtered[:i] > obj)
        after_are_worse = any_ax1_all(obj_values_filtered[i + 1 :] > obj)
        pareto_optimal[feasible_index] = before_are_worse and after_are_worse

    return pareto_optimal


class ParetoPlotBiObjective:
    """Plot a 2D Pareto front on a Matplotlib axes."""

    def __init__(
        self,
        axes: matplotlib.axes.Axes,
        obj_values: ndarray,
        pareto_optimal_loc: Sequence[bool],
        obj_names: Sequence[str],
        all_pareto_optimal: Sequence[bool],
        is_non_feasible: Sequence[bool],
        bi_obj: bool = False,
        show_non_feasible: bool = True,
    ) -> None:
        """
        Args:
            axes: A matplotlib axes on which to be plotted.
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
        self.__axes = axes
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
        self.__axes.set_xlabel(self.__obj_names[0])
        self.__axes.set_ylabel(self.__obj_names[1])

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
            self.__axes.scatter(
                self.__obj_values[self.__pareto_dominated_indexes, 0],
                self.__obj_values[self.__pareto_dominated_indexes, 1],
                color="b",
                label="Pareto dominated",
            )

    def __plot_non_feasible_points(self) -> None:
        """Plot the non-feasible points on the scatter plot."""
        if True in self.__is_non_feasible and self.__show_non_feasible:
            self.__axes.scatter(
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
            self.__axes.scatter(
                self.__obj_values[self.__pareto_optimal_all, 0],
                self.__obj_values[self.__pareto_optimal_all, 1],
                color="g",
                label=label,
            )

    def __plot_locally_pareto_optimal_points(self) -> None:
        """Plot the locally optimal Pareto points on the scatter plot."""
        if True in self.__pareto_optimal_loc and not self.__bi_objective:
            self.__axes.scatter(
                self.__obj_values[self.__pareto_optimal_loc, 0],
                self.__obj_values[self.__pareto_optimal_loc, 1],
                color="r",
                label="Locally Pareto optimal",
            )


def generate_pareto_plots(
    obj_values: ndarray,
    obj_names: Sequence[str],
    fig_size: tuple[float, float] = (10.0, 10.0),
    non_feasible_samples: ndarray | None = None,
    show_non_feasible: bool = True,
) -> matplotlib.figure.Figure:
    """Plot a 2D Pareto front.

    Args:
        obj_values: The objective function array of size (n_samples, n_objs).
        obj_names: The names of the objectives.
        fig_size: The matplotlib figure sizes in x and y directions, in inches.
        non_feasible_samples: The array of bool of size n_samples,
            True if the current sample is non-feasible.
            If None, all the samples are considered feasible.
        show_non_feasible: If True, show the non-feasible points in
            the Pareto front plot.

    Raises:
        ValueError: If the number of objective values and names are different.
    """
    n_obj = obj_values.shape[1]
    nb_pareto_pts = obj_values.shape[0]
    obj_names_length = len(obj_names)
    if obj_names_length != n_obj:
        msg = "Inconsistent objective values size and objective names: {} != {}".format(
            n_obj, obj_names_length
        )
        raise ValueError(msg)

    # If non_feasible_samples set to None,
    # then all the points are considered as feasible
    if non_feasible_samples is None:
        non_feasible_samples = full(nb_pareto_pts, False)
    is_feasible = ~non_feasible_samples

    pareto_opt_all = compute_pareto_optimal_points(obj_values, is_feasible)

    fig, axes = plt.subplots(n_obj - 1, n_obj - 1, figsize=fig_size, squeeze=False)
    fig.suptitle("Pareto front")

    # 0 vs 1   0 vs 2    0 vs 3
    #          1 vs 2    1 vs 3
    #                    2 vs 3

    # i j+1     i=0...nobj-1
    #           j=1....nobj

    for i, j in combinations(range(n_obj), 2):  # no duplication, j!=j
        obj_loc = vstack((obj_values[:, i], obj_values[:, j])).T
        pareto_opt_loc = compute_pareto_optimal_points(obj_loc, is_feasible)

        axe = axes[i, j - 1]
        bi_obj = True if n_obj == 2 else False
        plot_pareto_bi_obj = ParetoPlotBiObjective(
            axe,
            obj_loc,
            pareto_opt_loc,
            [obj_names[i], obj_names[j]],
            pareto_opt_all,
            non_feasible_samples,
            bi_obj=bi_obj,
            show_non_feasible=show_non_feasible,
        )
        plot_pareto_bi_obj.plot_on_axes()

        if i != j - 1:
            axes[j - 1, i].remove()

    # Ensure the unicity of the labels in the legend
    new_handles = []
    new_labels = []
    for axe in axes.flatten():
        handles, labels = axe.get_legend_handles_labels()
        for label, handle in zip(labels, handles):
            if label not in new_labels:
                new_labels.append(label)
                new_handles.append(handle)
    fig.legend(new_handles, new_labels, loc="lower left")

    return fig
