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
"""Services for handling Matplotlib figures, e.g. save and show."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_show_figure(
    fig: Figure,
    show: bool,
    file_path: str | Path,
    fig_size: tuple[float, float] | None = None,
) -> None:
    """Save or show a Matplotlib figure.

    Args:
        fig: The Matplotlib figure to be saved or shown.
        show: Whether to display the Matplotlib figure.
        file_path: The file path to save the Matplotlib figure.
            If ``None``, do not save the figure.
        fig_size: The width and height of the figure in inches, e.g. ``(w, h)``.
            If ``None``, use the current size of the figure.
    """
    save = file_path is not None

    if fig_size is not None:
        fig.set_size_inches(fig_size)

    if save:
        fig.savefig(str(file_path), bbox_inches="tight")

    if show:
        plt.show()

    if save:
        plt.close(fig)
