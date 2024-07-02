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
"""Test the function that save and/or show a Matplotlib figure."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from matplotlib import pyplot as plt
from matplotlib import rcParams

from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.matplotlib_figure import save_show_figure_from_file_path_manager


@pytest.mark.parametrize("file_path", ["", "file_name.pdf"])
@pytest.mark.parametrize("show", [True, False])
@pytest.mark.parametrize("fig_size", [(10, 10), None])
@pytest.mark.parametrize("use_save_show_figure", [True, False])
def test_save_show_figure(tmp_wd, use_save_show_figure, file_path, show, fig_size):
    """Verify that a Matplotlib figure is correctly saved."""
    fig, _ = plt.subplots()

    file_path_manager = FilePathManager(FilePathManager.FileType.FIGURE)
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.show"):
        if use_save_show_figure:
            save_show_figure(fig, show, file_path, fig_size=fig_size)
        else:
            save_show_figure_from_file_path_manager(
                fig,
                file_path_manager,
                show=show,
                file_path=file_path,
                fig_size=fig_size,
            )

        if fig_size is None:
            fig_size = rcParams["figure.figsize"]

        assert (fig.get_size_inches() == fig_size).all()

    if file_path is not None:
        assert Path(file_path).exists()

    plt.fignum_exists(fig.number)
