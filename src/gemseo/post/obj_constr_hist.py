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
#        :author: Pierre-Jean Barjhoux
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""History of the maximum constraint and objective value."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import SymmetricalLogLocator

from gemseo.post.base_post import BasePost
from gemseo.post.core.colormaps import RG_SEISMIC
from gemseo.post.obj_constr_hist_settings import ObjConstrHist_Settings

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.typing import NumberArray


class ObjConstrHist(BasePost[ObjConstrHist_Settings]):
    """History of the maximum constraint and objective value.

    The objective history is plotted with a line
    over the maximum constraint history plotted with the green-white-red color bar:

    - white: the constraint is active;
    - green: the equality constraint is violated while the inequality one is satisfied;
    - red: the inequality constraint is violated.
    """

    Settings: ClassVar[type[ObjConstrHist_Settings]] = ObjConstrHist_Settings

    __Y_MARGIN: Final[float] = 0.05
    """The left and right margin for the y-axis."""

    def _plot(self, settings: ObjConstrHist_Settings) -> None:
        constraint_names = settings.constraint_names

        # 0. Initialize the figure.
        fig, ax1 = plt.subplots(1, 1, figsize=settings.fig_size)
        n_iterations = len(self.database)
        ax1.set_xticks(range(n_iterations))
        ax1.set_xticklabels(map(str, range(1, n_iterations + 1)))
        mng = plt.get_current_fig_manager()
        assert mng is not None
        mng.resize(700, 1000)

        # 1. Plot the objective history versus the iterations with a curve.
        problem = self.optimization_problem
        objective_name = problem.standardized_objective_name
        obj_history, x_history = self.database.get_function_history(
            objective_name, with_x_vect=True
        )
        obj_history, x_history = np.array(obj_history).real, np.array(x_history).real
        if not problem.minimize_objective and not problem.use_standardized_objective:
            # Use the opposite of the standardized history.
            obj_history = -obj_history
            # Remove the minus sign prefixing the objective name.
            objective_name = objective_name[1:]

        obj_min, obj_max = obj_history.min(), obj_history.max()
        plt.plot(obj_history)
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel(objective_name, fontsize=12)
        margin = (obj_max - obj_min) * self.__Y_MARGIN
        plt.ylim([obj_min - margin, obj_max + margin])
        plt.grid(True)
        plt.title("Evolution of the objective and maximum constraint")

        # 2. Plot the maximum constraint history versus the iterations
        #    with green-white-red color map.
        # 2.a. Get inequality and equality constraint histories.
        ineq_history, ineq_names = self.__get_constraints(
            problem.constraints.get_inequality_constraints(), constraint_names
        )
        eq_history, eq_names = self.__get_constraints(
            problem.constraints.get_equality_constraints(), constraint_names
        )
        # 2.b. Concatenate the inequality and equality constraint histories.
        #      NB: Take absolute values of equality constraints for color map.
        constraint_history = np.concatenate(
            [
                constraint_history
                for constraint_history in [ineq_history, np.abs(eq_history)]
                if constraint_history.size > 0
            ],
            axis=1,
        )
        c_max = abs(constraint_history).max()
        im1 = ax1.imshow(
            np.atleast_2d(np.amax(constraint_history, axis=1)),
            cmap=RG_SEISMIC,
            interpolation="nearest",
            aspect="auto",
            extent=(-0.5, n_iterations - 0.5, obj_min - margin, obj_max + margin),
            norm=SymLogNorm(vmin=-c_max, vmax=c_max, linthresh=1.0),
        )
        # 2.c. Add vertical labels with constraint violation information.
        constraint_names = ineq_names + eq_names
        constraint_values = np.concatenate(
            [values for values in [ineq_history, eq_history] if values.size > 0], axis=1
        )
        ordinate = (obj_max + obj_min) / 2
        for iteration, i in enumerate(np.argmax(constraint_history, axis=1)):
            constraint_name = LogFormatterSciNotation().format_data(
                constraint_values[iteration, i]
            )
            text = ax1.text(
                iteration + 0.05,
                ordinate,
                f"${constraint_names[i]}={constraint_name}$",
                rotation="vertical",
                va="center",
            )
            text.set_bbox({"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})

        ax1.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        # 2.d. Add color map.
        col_bar = fig.colorbar(
            im1,
            ticks=SymmetricalLogLocator(linthresh=1.0, base=10),
            format=LogFormatterSciNotation(),
        )
        col_bar.ax.tick_params(labelsize=9)
        fig.tight_layout()
        self._add_figure(fig)

    def __get_constraints(
        self,
        constraints: Iterable[MDOFunction],
        all_constraint_names: Sequence[str],
    ) -> tuple[NumberArray, list[str]]:
        """Return the constraints with formatted shape.

        Args:
            constraints: The different constraints.
            all_constraint_names: The names of the constraints.
                If empty, use all the constraints.

        Returns:
            The history and the names of constraints.
        """
        constraint_names = []
        for constraint in constraints:
            if not all_constraint_names or constraint.name in all_constraint_names:
                constraint_names.append(constraint.name)  # noqa: PERF401

        if constraint_names:
            constraint_history, constraint_names, _ = self.database.get_history_array(
                function_names=constraint_names, with_x_vect=False
            )
        else:
            constraint_history, constraint_names = np.array([]), []

        # harmonization of tables format because constraints can be vectorial
        # or scalars. *vals.shape[0] = iteration, *vals.shape[1] = cstr values
        constraint_history = np.atleast_3d(constraint_history)
        shape = constraint_history.shape
        constraint_history = np.reshape(
            constraint_history, (shape[0], shape[1] * shape[2])
        )
        return constraint_history, constraint_names
