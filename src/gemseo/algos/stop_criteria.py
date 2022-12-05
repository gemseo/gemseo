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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Various termination criteria for drivers."""
from __future__ import annotations

from numpy import all
from numpy import allclose
from numpy import average


class TerminationCriterion(Exception):
    """Stop driver for some reason."""


class FunctionIsNan(TerminationCriterion):
    """Stops driver when a function has NaN value or NaN Jacobian."""


class DesvarIsNan(TerminationCriterion):
    """Stops driver when the design variables are nan."""


class MaxIterReachedException(TerminationCriterion):
    """Exception raised when the maximum number of iterations is reached."""


class MaxTimeReached(TerminationCriterion):
    """Exception raised when the maximum execution time is reached."""


class FtolReached(TerminationCriterion):
    """Exception raised when the f_tol_rel or f_tol_abs criteria is reached."""


class XtolReached(TerminationCriterion):
    """Exception raised when the x_tol_rel or x_tol_abs criteria is reached."""


def is_x_tol_reached(opt_problem, x_tol_rel=1e-6, x_tol_abs=1e-6, n_x=2):
    """Tests if the tolerance on the design variables are reached.

    The coordinate wise
    average of the last n_x points are taken Then it is checked that all points are
    within the distance of the center with relative and absolute tolerances specified by
    the user.

    Parameters
    ----------
    opt_problem: OptimizationProblem
        the optimization problem containing the iterations
    x_tol_rel: float
        relative tolerance
    x_tol_abs: float
        absolute tolerance
    n_x: int
        number of design vectors to account for
    """
    database = opt_problem.database
    if len(database) < n_x:
        return False

    x_values = database.get_last_n_x(n_x)

    # Checks that there is at least one feasible point
    if not any(opt_problem.is_point_feasible(database[x_val]) for x_val in x_values):
        return False

    x_average = average(x_values, axis=0)
    return all(
        [
            allclose(x_val, x_average, atol=x_tol_abs, rtol=x_tol_rel)
            for x_val in x_values
        ]
    )


def is_f_tol_reached(opt_problem, f_tol_rel=1e-6, f_tol_abs=1e-6, n_x=2):
    """Tests if the tolerance on the objective function are reached.

    The average function
    value of the last n_x points are taken Then it is checked that all points are within
    the distance of the center with relative and absolute tolerances specified by the
    user.

    Parameters
    ----------
    opt_problem: OptimizationProblem
        the optimization problem containing the iterations
    f_tol_rel: float
        relative tolerance
    f_tol_abs: float
        absolute tolerance
    n_x: int
        number of design vectors to account for
    """
    database = opt_problem.database
    if len(database) < n_x:
        return False

    # Checks that there is at least one feasible point
    x_values = database.get_last_n_x(n_x)
    if not any(opt_problem.is_point_feasible(database[x_val]) for x_val in x_values):
        return False

    obj_name = opt_problem.objective.name
    f_values = [
        f_value
        for f_value in [database.get_f_of_x(obj_name, x_val) for x_val in x_values]
        if f_value is not None
    ]
    if not f_values:
        return False

    f_average = average(f_values)
    return all(
        [
            allclose(f_val, f_average, atol=f_tol_abs, rtol=f_tol_rel)
            for f_val in f_values
        ]
    )
