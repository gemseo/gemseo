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
"""Generic tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemseo import execute_post


@pytest.mark.parametrize(
    ("class_name", "kwargs", "indices"),
    [
        ("OptHistoryView", {}, []),
        ("ObjConstrHist", {}, []),
        ("ConstraintsHistory", {"constraint_names": ["g_1"]}, [6, 7]),
    ],
)
def test_xticks(class_name, kwargs, indices):
    """Test that the xticks are sufficiently spaced (typically by using MaxNLocator).

    This is important when the iterations are represented on the x-axis
    as their number can be important.

    Args:
        class_name: The class name of the post-processor.
        kwargs: The parameters of the post-processor.
        indices: The indices of the axes having x-tick labels.
            If empty, assume that all the axes have x-tick labels.
    """
    post = execute_post(
        Path(__file__).parent / "sobieski_doe_20.hdf5",
        class_name,
        save=False,
        show=False,
        **kwargs,
    )
    for i, figure in enumerate(post.figures.values()):
        assert (figure.axes[0].get_xticks() == list(range(-2, 22, 2))).all()
        xticklabels = [text.get_text() for text in figure.axes[0].get_xticklabels()]
        if indices and i not in indices:
            assert xticklabels == []
        else:
            assert xticklabels == [""] + [str(i) for i in range(1, 20, 2)] + [""]
