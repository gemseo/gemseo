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
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from numpy import all as np_all
from numpy import any as np_any
from numpy import full
from numpy import ndarray
from numpy import vstack

from gemseo.algos.pareto.pareto_plot_biobjective import ParetoPlotBiObjective

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from gemseo.utils.matplotlib_figure import FigSizeType


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


def generate_pareto_plots(
    obj_values: ndarray,
    obj_names: Sequence[str],
    fig_size: FigSizeType = (10.0, 10.0),
    non_feasible_samples: ndarray | None = None,
    show_non_feasible: bool = True,
) -> Figure:
    """Plot a 2D Pareto front.

    Args:
        obj_values: The objective function array of size (n_samples, n_objs).
        obj_names: The names of the objectives.
        fig_size: The matplotlib figure sizes in x and y directions, in inches.
        non_feasible_samples: The array of bool of size n_samples,
            True if the current sample is non-feasible.
            If ``None``, all the samples are considered feasible.
        show_non_feasible: If ``True``, show the non-feasible points in
            the Pareto front plot.

    Raises:
        ValueError: If the number of objective values and names are different.
    """
    n_obj = obj_values.shape[1]
    nb_pareto_pts = obj_values.shape[0]
    obj_names_length = len(obj_names)
    if obj_names_length != n_obj:
        msg = (
            f"Inconsistent objective values size and objective names: "
            f"{n_obj} != {obj_names_length}"
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
        bi_obj = n_obj == 2
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
