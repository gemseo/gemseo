# -*- coding: utf-8 -*-
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

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gemseo.utils.py23_compat import Path


def save_show_figure(
    fig,  # type: Figure
    show,  # type:bool
    file_path,  # type: Union[str,Path]
    fig_size=None,  # type: Optional[Tuple[float, float]]
):  # type: (...) -> None
    """Save or show a Matplotlib figure.

    Args:
        fig: The Matplotlib figure to be saved or shown.
        show: If True, display the Matplotlib figure.
        file_path: The file path to save the Matplotlib figure.
            If None, do not save the figure.
        fig_size: The width and height of the figure in inches, e.g. `(w, h)`.
            If None, use the current size of the figure.
    """
    save = file_path is not None

    if save:
        if fig_size is not None:
            fig.set_size_inches(fig_size)
        fig.savefig(str(file_path), bbox_inches="tight")

    if show:
        try:
            plt.show(fig)
        except TypeError:
            plt.show()

    if save or show:
        plt.close(fig)
