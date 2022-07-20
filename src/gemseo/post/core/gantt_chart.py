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

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.file_path_manager import FileType
from gemseo.utils.matplotlib_figure import save_show_figure

DEFAULT_NAME = "gantt_chart"


def create_gantt_chart(
    file_path: str | Path = DEFAULT_NAME,
    save: bool = True,
    show: bool = False,
    file_extension: str | None = None,
    fig_size: tuple[float, float] = (15.0, 10.0),
    font_size: int = 12,
    disc_names: Sequence[str] | None = None,
) -> plt.Figure:
    """Generate a gantt chart with processes execution time data.

    The disciplines names are used as labels and plotted on rows.
    The x labels are the execution start and end times of all disciplines.

    Both executions and linearizations times are plotted.

    Warning:
        ``MDODiscipline.activate_time_stamps()`` must be called first.

    Args:
        file_path: The path to save the figure.
        save: Whether to save the figure.
        show: Whether to show the figure.
        file_extension: A file extension, e.g. 'png', 'pdf', 'svg', ...
            If ``None``, use the default file extension.
        fig_size: The figure size.
        font_size: The size of the fonts in the plot.
        disc_names: The names of the disciplines to plot.
            If ``None``, plot all the disciplines for which time stamps exist.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If the time stamps is not activated or if a discipline has no
            time stamps.
    """
    time_stamps = MDODiscipline.time_stamps
    if time_stamps is None:
        raise ValueError("Time stamps are not activated in MDODiscipline")

    fig, ax = plt.subplots(figsize=fig_size)

    if disc_names is None:
        disc_names = list(time_stamps.keys())
    else:
        missing = list(set(disc_names) - set(time_stamps.keys()))
        if missing:
            raise ValueError(f"The disciplines: {missing}, have no time stamps.")

    ax.set_ylim(5, 10 * len(disc_names) + 15)
    ax.set_yticklabels(disc_names)
    ax.set_yticks([5 + (i + 1) * 10 for i in range(len(disc_names))])
    ax.set_xlabel("Time")
    ax.set_ylabel("Disciplines")
    ax.grid(True)

    # Minimum time as a reference
    min_t = min(stamps[0][0] for stamps in time_stamps.values())

    # Blue for execution, red for linearization
    colors = {False: "tab:blue", True: "tab:red"}
    for i, name in enumerate(disc_names):
        stamps_orig = time_stamps[name]
        stamps = [(l - min_t, u - l) for (l, u, _) in stamps_orig]
        facecolors = [colors[s[2]] for s in stamps_orig]
        ax.broken_barh(stamps, ((i + 1) * 10, 9), facecolors=facecolors)

    # Set all fonts sizes
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(font_size)

    if save:
        file_path = FilePathManager(
            FileType.FIGURE, default_name=DEFAULT_NAME
        ).create_file_path(file_path=file_path, file_extension=file_extension)
    else:
        file_path = None

    save_show_figure(fig, show, file_path)

    return fig
