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
from gemseo.utils.matplotlib_figure import save_show_figure
from matplotlib import pyplot as plt
from matplotlib import rcParams


@pytest.mark.parametrize("file_path", [None, "file_name.pdf"])
@pytest.mark.parametrize("show", [True, False])
@pytest.mark.parametrize("fig_size", [(10, 10), None])
def test_process(tmp_wd, pyplot_close_all, file_path, show, fig_size):
    """Verify that a Matplotlib figure is correctly saved."""
    fig, axes = plt.subplots()

    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.show"):
        save_show_figure(fig, show, file_path, fig_size)
        if fig_size is None:
            fig_size = rcParams["figure.figsize"]
        assert (fig.get_size_inches() == fig_size).all()

    if file_path is not None:
        assert Path(file_path).exists()

    plt.fignum_exists(fig.number)
