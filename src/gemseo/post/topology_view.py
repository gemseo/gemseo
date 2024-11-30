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
from typing import ClassVar
from typing import cast

from matplotlib import colors
from matplotlib import pyplot as plt

from gemseo.post.base_post import BasePost
from gemseo.post.topology_view_settings import TopologyView_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class TopologyView(BasePost[TopologyView_Settings]):
    """Visualization of the solution of a 2D topology optimization problem."""

    Settings: ClassVar[type[TopologyView_Settings]] = TopologyView_Settings

    def _plot(
        self,
        settings: TopologyView_Settings,
    ) -> None:
        iterations = settings.iterations
        observable = settings.observable
        n_x = settings.n_x
        n_y = settings.n_y

        if isinstance(iterations, int):
            iterations = [iterations]
        elif not iterations:
            iterations = [len(self.database)]

        for iteration in iterations:
            plt.ion()  # Ensure that redrawing is possible
            design = self.database.get_x_vect(iteration)
            fig, ax = plt.subplots()
            if observable:
                data = (
                    -cast(
                        "RealArray",
                        self.database.get_function_value(observable, design),
                    )
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
            fig.tight_layout()
            self._add_figure(fig, f"configuration_{iteration}")
