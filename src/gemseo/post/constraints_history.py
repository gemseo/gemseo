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
"""A matrix of constraint history plots."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from matplotlib import colormaps
from matplotlib import pyplot
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import MaxNLocator
from numpy import abs as np_abs
from numpy import arange
from numpy import atleast_2d
from numpy import atleast_3d
from numpy import diff
from numpy import flip
from numpy import interp
from numpy import linspace
from numpy import max as np_max
from numpy import sign

from gemseo.post._engine.colormap import RG_SEISMIC
from gemseo.post.constraints_history_settings import ConstraintsHistory_Settings
from gemseo.post.core.base_post import BasePost

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


_BAND_OPACITY: Final[float] = 0.6
"""The opacity of the background color bands."""


def _fade(colormap: Colormap) -> ListedColormap:
    """Fade a colormap to white.

    The bands are drawn opaque rather than with an ``alpha``, so that their colors
    do not depend on what is behind them, e.g. the background of a web page seen
    through a figure saved with a transparent background.

    Args:
        colormap: The colormap to fade.

    Returns:
        The colormap blended with white, as if drawn with an opacity of
        [_BAND_OPACITY][gemseo.post.constraints_history._BAND_OPACITY].
    """
    colors = colormap(linspace(0.0, 1.0, colormap.N))
    return ListedColormap(_BAND_OPACITY * colors + 1.0 - _BAND_OPACITY)


_EQUALITY_COLORMAP: Final[ListedColormap] = _fade(colormaps["seismic"])
"""The colormap of the background color bands of the equality constraints."""

_INEQUALITY_COLORMAP: Final[ListedColormap] = _fade(RG_SEISMIC)
"""The colormap of the background color bands of the inequality constraints."""


class ConstraintsHistory(BasePost[ConstraintsHistory_Settings]):
    r"""A matrix of constraint history plots.

    A blue line represents the values of a constraint w.r.t. the iterations.

    A background color indicates whether the constraint is satisfied (green), active
    (white) or violated (red). This band is opaque, so that its colors do not depend
    on what is behind the figure.

    A horizontal line indicates the value for which an inequality constraint is
    active or an equality constraint is satisfied, namely $0$. A horizontal
    dashed line indicates the value below which an inequality constraint is satisfied
    *with a tolerance level*, namely $\varepsilon$.

    For an equality constraint, the horizontal dashed lines indicate the values
    between which the constraint is satisfied *with a tolerance level*, namely
    $-\varepsilon$ and $\varepsilon$.

    A vertical line indicates the last iteration (or pseudo-iteration) where the
    constraint is (or should be) active.

    These lines are drawn in the color of the axes, namely the matplotlib rcParam
    ``axes.edgecolor``.
    """

    settings_class: ClassVar[type[ConstraintsHistory_Settings]] = (
        ConstraintsHistory_Settings
    )

    def _plot(self, settings: ConstraintsHistory_Settings) -> None:
        """
        Raises:
            ValueError: When an item of `constraint_names` is not a constraint name.
        """  # noqa: D205, D212, D415
        constraint_names = settings.constraint_names
        line_style = settings.line_style
        add_points = settings.add_points
        all_constraint_names = (
            self._optimization_metadata.output_name_to_constraint_names.keys()
        )
        for constraint_name in constraint_names:
            if constraint_name not in all_constraint_names:
                msg = (
                    "Cannot build constraints history plot, "
                    f"{constraint_name} is not a constraint name."
                )
                raise ValueError(msg)

        constraint_names = [
            item
            for variable in constraint_names
            for item in (
                self._optimization_metadata.output_name_to_constraint_names[variable]
            )
        ]

        constraint_histories = (
            self._dataset.get_view(variable_names=constraint_names).dropna().to_numpy()
        )
        constraint_names = self._dataset.get_columns(constraint_names)

        # harmonization of tables format because constraints can be vectorial
        # or scalars. *vals.shape[0] = iteration, *vals.shape[1] = cstr values
        constraint_histories = atleast_3d(constraint_histories)
        constraint_histories = constraint_histories.reshape((
            constraint_histories.shape[0],
            constraint_histories.shape[1] * constraint_histories.shape[2],
        ))

        # prepare the main window
        fig, axs = pyplot.subplots(
            nrows=ceil(len(constraint_names) / 2),
            ncols=2,
            sharex=True,
            figsize=settings.fig_size,
        )

        fig.suptitle("Evolution of the constraints w.r.t. iterations", fontsize=14)

        iterations = arange(len(constraint_histories))
        n_iterations = len(iterations)
        eq_constraint_names = self._dataset.equality_constraint_names

        # for each subplot
        for constraint_history, constraint_name, ax in zip(
            constraint_histories.T, constraint_names, axs.ravel(), strict=False
        ):
            f_name = constraint_name.split("[")[0]
            is_eq_constraint = f_name in eq_constraint_names
            if is_eq_constraint:
                cmap = _EQUALITY_COLORMAP
                constraint_type = "equality"
                tolerance = self._optimization_metadata.tolerances.equality
            else:
                cmap = _INEQUALITY_COLORMAP
                constraint_type = "inequality"
                tolerance = self._optimization_metadata.tolerances.inequality

            # prepare the graph
            ax.grid(True)
            ax.set_title(f"{constraint_name} ({constraint_type})")
            ax.set_xticks(range(n_iterations))
            ax.set_xticklabels(range(1, n_iterations + 1))
            ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
            axis_color = rcParams["axes.edgecolor"]
            ax.axhline(tolerance, color=axis_color, linestyle="--")
            ax.axhline(0.0, color=axis_color)
            if is_eq_constraint:
                ax.axhline(-tolerance, color=axis_color, linestyle="--")

            # Add line and points
            ax.plot(iterations, constraint_history, linestyle=line_style)
            if add_points:
                ax.scatter(iterations, constraint_history)

            # Plot color bars
            maximum = np_max(np_abs(constraint_history))
            margin = 2 * maximum * 0.05
            ax.imshow(
                atleast_2d(constraint_history),
                cmap=cmap,
                interpolation="nearest",
                aspect="auto",
                norm=SymLogNorm(vmin=-maximum, vmax=maximum, linthresh=1.0),
                extent=[-0.5, n_iterations - 0.5, -maximum - margin, maximum + margin],
            )

            # Plot a vertical line at the last iteration (or pseudo-iteration)
            # where the constraint is (or should be) active.
            indices_before_sign_change = diff(sign(constraint_history)).nonzero()[0]
            if indices_before_sign_change.size != 0:
                index_before_last_sign_change = indices_before_sign_change[-1]
                indices = [
                    index_before_last_sign_change,
                    index_before_last_sign_change + 1,
                ]
                constraint_values = constraint_history[indices]
                iteration_values = iterations[indices]
                if constraint_values[1] < constraint_values[0]:
                    constraint_values = flip(constraint_values)
                    iteration_values = flip(iteration_values)

                ax.axvline(
                    interp(0.0, constraint_values, iteration_values), color=axis_color
                )

        fig.tight_layout()
        self._add_figure(fig)
