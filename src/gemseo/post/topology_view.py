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
"""View of the solution of a 2D topology optimization problem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import colors
from matplotlib import pyplot as plt

from gemseo.post.opt_post_processor import OptPostProcessor

if TYPE_CHECKING:
    from collections.abc import Iterable


class TopologyView(OptPostProcessor):
    """Visualization of the solution of a 2D topology optimization problem."""

    DEFAULT_FIG_SIZE = (11.0, 6.0)

    def _plot(
        self,
        n_x: int,
        n_y: int,
        observable: str | None = None,
        iterations: int | Iterable[int] | None = None,
    ) -> None:
        """Plot the design variable or an observable field patch plot.

        Args:
            n_x: The number of elements in the horizontal direction.
            n_y: The number of elements in the vertical direction.
            observable: The name of the observable to be plotted.
                It should be of size ``n_x*n_y``.
            iterations: The iterations of the optimization history.
                If ``None``, the last iteration is taken.
        """
        if iterations is None:
            iterations = [len(self.database)]
        elif isinstance(iterations, int):
            iterations = [iterations]
        for iteration in iterations:
            plt.ion()  # Ensure that redrawing is possible
            design = self.database.get_x_vect(iteration)
            fig, ax = plt.subplots()
            if observable:
                data = (
                    -self.database.get_function_value(observable, design)
                    .reshape((n_x, n_y))
                    .T
                )
            else:
                data = -design.reshape((n_x, n_y)).T

            im = ax.imshow(
                data,
                cmap="gray",
                interpolation="none",
                norm=colors.Normalize(vmin=-1, vmax=0),
            )
            im.set_array(data)
            plt.axis("off")
            self._add_figure(fig, f"configuration_{iteration}")
