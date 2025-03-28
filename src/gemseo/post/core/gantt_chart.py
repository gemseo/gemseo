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
#                         documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Generate a gantt chart with processes execution time data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.matplotlib_figure import FigSizeType
from gemseo.utils.matplotlib_figure import save_show_figure

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

DEFAULT_NAME = "gantt_chart"


def create_gantt_chart(
    file_path: str | Path = DEFAULT_NAME,
    save: bool = True,
    show: bool = False,
    file_extension: str = "",
    fig_size: FigSizeType = (15.0, 10.0),
    font_size: int = 12,
    disc_names: Sequence[str] = (),
) -> plt.Figure:
    """Generate a gantt chart with processes execution time data.

    The disciplines names are used as labels and plotted on rows.
    The x labels are the execution start and end times of all disciplines.

    Both executions and linearizations times are plotted.

    .. warning::
        ``ExecutionStatistics.is_time_stamps_enabled = True`` must be done first.

    Args:
        file_path: The path to save the figure.
        save: Whether to save the figure.
        show: Whether to show the figure.
        file_extension: A file extension, e.g. 'png', 'pdf', 'svg', ...
            If empty, use the default file extension.
        fig_size: The figure size.
        font_size: The size of the fonts in the plot.
        disc_names: The names of the disciplines to plot.
            If empty, plot all the disciplines for which time stamps exist.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If the time stamps is not enabled or if a discipline has no
            time stamps.
    """
    time_stamps = ExecutionStatistics.time_stamps
    if time_stamps is None:
        msg = "Time stamps are not enabled in Discipline"
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=fig_size)

    if disc_names:
        missing = list(set(disc_names) - set(time_stamps.keys()))
        if missing:
            msg = f"The disciplines: {missing}, have no time stamps."
            raise ValueError(msg)
    else:
        disc_names = list(time_stamps.keys())

    ax.set_ylim(5, 10 * len(disc_names) + 15)
    ax.set_yticks([5 + (i + 1) * 10 for i in range(len(disc_names))])
    ax.set_yticklabels(disc_names)
    ax.set_xlabel("Time (s)")
    ax.set_title("Execution (blue) and linearization (red) of the disciplines")
    ax.grid(True)

    # Minimum time as a reference
    min_t = min(stamps[0][0] for stamps in time_stamps.values())

    # Blue for execution, red for linearization
    colors = {False: "tab:blue", True: "tab:red"}
    for i, name in enumerate(disc_names):
        stamps_orig = time_stamps[name]
        stamps = [(low - min_t, up - low) for (low, up, _) in stamps_orig]
        face_colors = [colors[s[2]] for s in stamps_orig]
        ax.broken_barh(stamps, ((i + 1) * 10, 9), facecolors=face_colors)

    # Set all fonts sizes
    for item in {
        ax.title,
        ax.xaxis.label,
        ax.yaxis.label,
        *ax.get_xticklabels(),
        *ax.get_yticklabels(),
    }:
        item.set_fontsize(font_size)

    if save:
        file_path = FilePathManager(
            FilePathManager.FileType.FIGURE, default_name=DEFAULT_NAME
        ).create_file_path(file_path=file_path, file_extension=file_extension)
    else:
        file_path = ""

    save_show_figure(fig, show, file_path)

    return fig
