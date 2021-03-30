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
Compute and display a Pareto Front
**********************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import range, str
from itertools import combinations

import matplotlib.pyplot as plt
from future import standard_library
from numpy import all as np_all
from numpy import any as np_any
from numpy import full, vstack

standard_library.install_aliases()


def select_pareto_optimal(obj_values):
    """
    Compute the Pareto front
    Search for all non dominated points, ie there exists j such that
    there is no lower value for obj_values[:,j] that does not degrade
    at least one other objective  obj_values[:,i]

    :param obj_values: objective function array, of size (n_samples, n_objs)
    :returns pareto_optimal: vector of booleans of size n_samples, True if
        Pareto optimal
    """
    pareto_optimal = full(obj_values.shape[0], True)

    def any_ax1_all(arr):
        return np_all(np_any(arr, axis=1))

    for i, obj in enumerate(obj_values):
        before_are_worse = any_ax1_all(obj_values[:i] > obj)
        after_are_worse = any_ax1_all(obj_values[i + 1 :] > obj)
        pareto_optimal[i] = before_are_worse and after_are_worse
    return pareto_optimal


def plot_pareto_bi_obj(axe, obj_values, pareto_optimal, obj_names, all_pareto=None):
    """
    Plot a 2D Pareto front

    :param axe: matplotlib axe on which to be plotted
    :param obj_values: objective function array, of size (n_samples, n_objs)
    :param pareto_optimal: vector of booleans of size n_samples,
        True if Pareto optimal.
    :param obj_names: names of the objectives
    :param all_pareto: indices of points that are pareto optimal wrt all
        criteria.
    """
    axe.scatter(obj_values[:, 0], obj_values[:, 1], color="b")

    if all_pareto is not None:
        axe.scatter(obj_values[all_pareto, 0], obj_values[all_pareto, 1], color="g")

    axe.scatter(obj_values[pareto_optimal, 0], obj_values[pareto_optimal, 1], color="r")

    axe.set_xlabel(obj_names[0])
    axe.set_ylabel(obj_names[1])


def generate_pareto_plots(obj_values, obj_names, figsize=(10, 10)):
    """
    Plot a 2D Pareto front
    :param  obj_values: objective function array, of size (n_samples, n_objs)
    :param obj_names: names of the objectives
    :param figsize: matplotlib figure size in inches
    """
    n_obj = obj_values.shape[1]
    if len(obj_names) != n_obj:
        raise ValueError(
            "Inconsistent objective values size and objective names!"
            + str(n_obj)
            + " !="
            + str(len(obj_names))
        )

    fig, axes = plt.subplots(n_obj - 1, n_obj - 1, figsize=figsize)

    fig.suptitle("Pareto front")

    pareto_opt_all = select_pareto_optimal(obj_values)
    # 0 vs 1   0 vs 2    0 vs 3
    #          1 vs 2    1 vs 3
    #                    2 vs 3

    # i j+1     i=0...nobj-1
    #           j=1....nobj

    for i, j in combinations(range(n_obj), 2):  # no duplication, j!=j
        if i == n_obj - 1:
            continue

        obj_loc = vstack((obj_values[:, i], obj_values[:, j])).T

        pareto_opt_loc = select_pareto_optimal(obj_loc)
        if n_obj == 2:
            axe = axes
        else:
            axe = axes[i, j - 1]
        plot_pareto_bi_obj(
            axe, obj_loc, pareto_opt_loc, [obj_names[i], obj_names[j]], pareto_opt_all
        )
        if i != j - 1:
            axes[j - 1, i].remove()

    return fig
