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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A mixin for the probability distribution of a scalar random variable."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Union

from matplotlib import pyplot as plt
from numpy import arange

from gemseo.utils.matplotlib_figure import save_show_figure

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution

StandardParametersType = Mapping[str, Union[str, int, float]]
ParametersType = Union[tuple[str, int, float], StandardParametersType]


class ScalarDistributionMixin:
    """A mixin for the probability distribution of a scalar random variable."""

    JOINT_DISTRIBUTION_CLASS: ClassVar[type[BaseJointDistribution]]
    """The class of the joint distribution associated with this distribution."""

    def plot(
        self,
        show: bool = True,
        save: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_extension: str = "",
    ) -> Figure:
        """Plot both probability and cumulative density functions.

        Args:
            save: Whether to save the figure.
            show: Whether to display the figure.
            file_path: The path of the file to save the figure.
                If the extension is missing, use ``file_extension``.
                If empty,
                create a file path
                from ``directory_path``, ``file_name`` and ``file_extension``.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_extension: A file extension, e.g. ``'png'``, ``'pdf'``, ``'svg'``, ...
                If empty, use a default file extension.

        Returns:
            The figure.
        """
        l_b = self.num_lower_bound
        u_b = self.num_upper_bound
        x_values = arange(l_b, u_b, (u_b - l_b) / 100)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.2))
        fig.suptitle(repr(self))
        ax1.plot(x_values, [self._pdf(x_value) for x_value in x_values])
        ax1.grid()
        ax1.set_ylabel("Probability density function")
        ax1.set_box_aspect(1)
        ax2.plot(x_values, [self._cdf(x_value) for x_value in x_values])
        ax2.grid()
        ax2.set_ylabel("Cumulative distribution function")
        ax2.yaxis.tick_right()
        ax2.set_box_aspect(1)
        if save:
            file_path = self._file_path_manager.create_file_path(
                file_path=file_path,
                file_name=file_name,
                directory_path=directory_path,
                file_extension=file_extension,
            )
        else:
            file_path = ""

        save_show_figure(fig, show, file_path)
        return fig

    @abstractmethod
    def _pdf(
        self,
        value: float,
    ) -> float:
        """Probability density function (PDF).

        Args:
            value: An evaluation point.

        Returns:
            The PDF value at the evaluation point.
        """

    @abstractmethod
    def _cdf(
        self,
        level: float,
    ) -> float:
        """Cumulative distribution function (CDF).

        Args:
            level: A probability level.

        Returns:
            The CDF value for the probability level.
        """
