# Copyright 2022 Airbus SAS
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
#
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Gabriel Max DE MENDONÇA ABRANTES
"""Pareto front."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from numpy import all as np_all
from numpy import argwhere
from numpy import array
from numpy import concatenate as np_concat
from numpy import min as np_min
from numpy import zeros
from numpy.linalg import norm as np_norm
from pandas import DataFrame
from pandas import MultiIndex
from pandas import concat as pd_concat
from prettytable import PrettyTable

from gemseo.algos.pareto.utils import compute_pareto_optimal_points
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import _format_value_in_pretty_table_6
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo import OptimizationProblem
    from gemseo.typing import RealArray


@dataclass
class ParetoFront:
    """A Pareto front.

    The design and objective vectors are noted ``x`` and ``f`` respectively.
    """

    distance_from_utopia: float
    """The shortest Euclidean distance from the Pareto front to the
    :attr:`.f_utopia`."""

    f_anchors: RealArray
    """The values of the objectives of all anchor points.

    At those points, each objective is minimized one at a time.

    Its shape is ``(n_anchors, f_dimension)``.
    """

    f_anti_utopia: RealArray
    """The anti-utopia point, i.e. the maximum objective vector.

    Its shape is ``(f_dimension,)``.
    """

    f_optima: RealArray
    """The objective values of the Pareto optima.

    Its shape is ``(n_optima, f_dimension)``.
    """

    f_utopia: RealArray
    """The utopia point, i.e. the minimum objective vector.

    In most Pareto fronts, there is no design value for which the objective is equal to
    the utopia.

    Its shape is ``(f_dimension,)``.
    """

    f_utopia_neighbors: RealArray
    """The objectives value of the closest point(s) to the :attr:`.f_utopia`.

    The distance separating them from :attr:`.f_utopia` is
    :attr:`.distance_from_utopia`.

    Its shape is ``(n_neighbors, f_dimension)``.
    """

    x_anchors: RealArray
    """The values of the design variables values of all anchor points.

    At those points, each objective is minimized one at a time.

    Its shape is ``(n_anchors, x_dimension)``.
    """

    x_optima: RealArray
    """The values of the design variables of the Pareto optima.

    Its shape is ``(n_optima, x_dimension)``.
    """

    x_utopia_neighbors: RealArray
    """The design variables value of the closest point(s) to the :attr:`.f_utopia`.

    The distance separating them from :attr:`.f_utopia` is
    :attr:`.distance_from_utopia`.

    Its shape is ``(n_neighbors, x_dimension)``.
    """

    _anchors_neighbors: DataFrame = field(init=False)
    """Hold the points of interest to be shown in the optimization result."""

    _problem: OptimizationProblem = field(init=False)
    """The optimization problem associated to the Pareto front."""

    @classmethod
    def from_optimization_problem(cls, problem) -> ParetoFront:
        """Create a :class:`.ParetoFront` from an :class:`.OptimizationProblem`.

        Args:
            problem: The optimization problem.

        Returns:
            The Pareto front.
        """
        f_optima, x_optima = cls.__get_optima(problem)
        f_utopia = f_optima.min(axis=0)
        f_anti_utopia = f_optima.max(axis=0)
        f_anchors, x_anchors = cls.__get_anchors(f_optima, x_optima)
        f_utopia_neighbors, x_utopia_neighbors, distance_from_utopia = (
            cls.__get_utopia_nearest_neighbors(f_optima, x_optima, f_utopia)
        )

        # Get dataset as a dataframe.
        full_history = problem.to_dataset()

        # Get design variables group,
        # and reorder the columns to match the design space order.
        desvar_history = full_history.get_view(
            variable_names=problem.design_space.variable_names
        )
        ind_anchors = [
            full_history.index[np_all(desvar_history == x_anchor, axis=1)][0]
            for x_anchor in x_anchors
        ]
        ind_neighbors = [
            full_history.index[np_all(desvar_history == x_utopia_neighbor, axis=1)][0]
            for x_utopia_neighbor in x_utopia_neighbors
        ]

        # DataFrame with points of interest.
        anchors = full_history.loc[ind_anchors].droplevel(0, axis=1)
        anchors.index = [f"anchor_{i + 1}" for i in range(len(ind_anchors))]
        neighbors = full_history.loc[ind_neighbors].droplevel(0, axis=1)
        neighbors.index = [f"compromise_{i + 1}" for i in range(len(ind_neighbors))]
        anchors_neighbors = pd_concat([anchors, neighbors], axis=0)

        # Shift dimensions to start at 1.
        new_columns = [
            (*c[0:-1], str(int(c[-1]) + 1)) for c in anchors_neighbors.columns
        ]
        anchors_neighbors.columns = MultiIndex.from_tuples(new_columns)

        pareto_front = cls(
            f_optima=f_optima,
            x_optima=x_optima,
            f_utopia=f_utopia,
            f_anti_utopia=f_anti_utopia,
            f_anchors=f_anchors,
            x_anchors=x_anchors,
            f_utopia_neighbors=f_utopia_neighbors,
            x_utopia_neighbors=x_utopia_neighbors,
            distance_from_utopia=distance_from_utopia,
        )
        pareto_front._problem = problem
        pareto_front._anchors_neighbors = anchors_neighbors
        return pareto_front

    @staticmethod
    def __get_utopia_nearest_neighbors(
        f_optima: RealArray,
        x_optima: RealArray,
        f_utopia: RealArray,
    ) -> tuple[RealArray, RealArray, float]:
        """Get the utopia's nearest neighbors.

        Args:
            f_optima: The objective values of the Pareto optima.
            x_optima: The values of the design variables of the Pareto optima.
            f_utopia: The utopia point, i.e. the minimum objective vector.

        Returns:
            The objective values of the utopia's nearest neighbors.
            The values of the design variables of the utopia's nearest neighbors.
            The shortest Euclidean distance fron the Pareto front to the utopia.

        Raises:
            ValueError: If the utopia does not have the appropriate dimension.
        """
        if f_utopia.shape != (f_optima.shape[1],):
            msg = (
                f"Reference point {f_utopia} does not have the "
                "same amount of objectives as the pareto front."
            )
            raise ValueError(msg)

        distances = np_norm(f_optima - f_utopia, axis=1)
        min_distance = np_min(distances)
        min_indices = argwhere(distances == min_distance).flatten()
        return f_optima[min_indices], x_optima[min_indices], min_distance

    @staticmethod
    def __get_optima(
        problem: OptimizationProblem,
    ) -> tuple[RealArray, RealArray]:
        """Get the Pareto optima from the optimization history.

        A Pareto optimum is a non-dominated point.

        Args:
            problem: The optimization problem containing the optimization history.

        Returns:
            First the objectives' values of the Pareto optima,
            then the values of their design variables.
        """
        n_iter = len(problem.database)

        dv_history = zeros((n_iter, problem.design_space.dimension))
        obj_history = zeros((n_iter, problem.objective.dim))
        feasibility = zeros(n_iter)

        for iteration, item in enumerate(problem.database.items()):
            x_vect, out_val = item
            dv_history[iteration] = x_vect.unwrap()
            if problem.objective.name in out_val:
                obj_history[iteration] = array(out_val[problem.objective.name])
                feasibility[iteration] = problem.constraints.is_point_feasible(out_val)
            else:
                obj_history[iteration] = float("nan")
                feasibility[iteration] = False

        optimal_points = compute_pareto_optimal_points(obj_history, feasibility)
        return obj_history[optimal_points], dv_history[optimal_points]

    @staticmethod
    def __get_anchors(
        pareto_front: RealArray,
        pareto_set: RealArray,
    ) -> tuple[RealArray, RealArray]:
        """Get Pareto's anchor points.

        Args:
            pareto_front: The objectives' value of all non-dominated points.
            pareto_set: The design variables' value of all non-dominated points.

        Returns:
            The objectives' values of all anchor points.
            The design variables' values of all anchor points.
        """
        n_obj = pareto_front.shape[1]

        anchor_points_index = zeros(n_obj, dtype=int)
        min_pf = np_min(pareto_front, axis=0)
        for obj_i in range(n_obj):
            anchor_points_index[obj_i] = argwhere(
                pareto_front[:, obj_i] == min_pf[obj_i]
            )[0]

        return pareto_front[anchor_points_index], pareto_set[anchor_points_index]

    @staticmethod
    def __get_pretty_table_from_df(
        df: DataFrame,
    ) -> PrettyTable:
        """Build a tabular view of the Pareto problem.

        Args:
            df: The Pareto data.

        Returns:
            A tabular view of the Pareto problem.
        """
        fields = [df.index.name or "name"]
        if df.columns.nlevels == 1:
            fields += list(df.columns)
        else:
            fields += [f"{col[0]} ({pretty_str(col[1:])})" for col in df.columns]

        table = PrettyTable(fields)
        table.custom_format = _format_value_in_pretty_table_6
        for _, row in df.iterrows():
            name = row.name
            if isinstance(name, tuple):
                content = [f"{name[0]} ({pretty_str(name[1:])})"]
            else:
                content = [name]
            content += row.to_list()
            table.add_row(content)
        table.align = "r"
        return table

    def __str__(self) -> str:
        obj_names = [self._problem.standardized_objective_name]
        c_names = self._problem.constraints.get_names()
        dv_names = self._problem.design_space

        msg = MultiLineString()
        msg.add(
            "Pareto optimal points : {} / {}",
            self.f_optima.shape[0],
            len(self._problem.database),
        )
        msg.add("Utopia point : {}", self.f_utopia)
        msg.add("Compromise solution (closest to utopia) : {}", self.f_utopia_neighbors)
        msg.add("Distance from utopia : {}", self.distance_from_utopia)
        msg.add("Objective values:")
        msg.indent()
        for line in str(
            self.__get_pretty_table_from_df(self._anchors_neighbors[obj_names].T)
        ).split("\n"):
            msg.add("{}", line)
        if self._problem.constraints:
            msg.dedent()
            msg.add("Constraint values:")
            msg.indent()
            for line in str(
                self.__get_pretty_table_from_df(self._anchors_neighbors[c_names].T)
            ).split("\n"):
                msg.add("{}", line)
        msg.dedent()
        msg.add("Design space:")
        msg.indent()

        # Prepare DataFrame for design space.
        ds = self._problem.design_space
        cols = MultiIndex.from_tuples([
            (n, str(d + 1)) for n in ds for d in range(ds.get_size(n))
        ])

        types = np_concat([[ds.get_type(var)] * ds.get_size(var) for var in ds])

        df_lb = DataFrame(
            ds.get_lower_bounds().reshape(1, -1), columns=cols, index=["lower_bound"]
        )
        df_ub = DataFrame(
            ds.get_upper_bounds().reshape(1, -1), columns=cols, index=["upper_bound"]
        )
        df_types = DataFrame(types.reshape(1, -1), columns=cols, index=["type"])
        df_interest_dv = pd_concat([
            df_lb,
            self._anchors_neighbors[dv_names],
            df_ub,
            df_types,
        ])

        for line in str(self.__get_pretty_table_from_df(df_interest_dv.T)).split("\n"):
            msg.add("{}", line)
        return str(msg)
