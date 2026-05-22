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
"""Test the class Surfaces plotting samples of a 2D variable with surfaces."""

from __future__ import annotations

import pytest
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.surfaces import Surfaces
from gemseo.post.dataset.surfaces_settings import Surfaces_Settings


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing two samples of the 2D variable 'output'.

    This variable is plotted over the 2D mesh: [[0, 0], [0, 1], [1, 0], [1, 1]].

    The samples are [0., 1., 1. , 0.] and [1., 0., 0., 1.].
    """
    sample1 = [0.0, 1.0, 1.0, 0.0]
    sample2 = [1.0, 0.0, 0.0, 1.0]
    data_array = array([sample1, sample2])
    dataset = Dataset.from_array(
        data_array,
        variable_names=["output"],
        variable_name_to_n_components={"output": 4},
    )
    dataset.misc["mesh"] = array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    return dataset


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"samples": [1]},
        {"add_points": True},
        {"fill": False},
        {"levels": 2},
        {
            "properties": {
                "xlabel": "The xlabel",
                "ylabel": "The ylabel",
                "title": "The title",
            }
        },
    ],
)
def test_plot(kwargs, dataset, snapshot_matplotlib) -> None:
    """Test images created by Surfaces._plot against references."""
    properties = kwargs.pop("properties", {})
    settings = Surfaces_Settings(mesh="mesh", variable="output", **kwargs, **properties)
    plot = Surfaces(dataset, settings)
    plot.execute(save=False)


def test_save_multiple_files(dataset, tmp_wd) -> None:
    """Check that the generation of multiple files is OK."""
    settings = Surfaces_Settings(mesh="mesh", variable="output")
    surfaces = Surfaces(dataset, settings)
    surfaces.execute()
    file_paths = [tmp_wd / "surfaces_0.png", tmp_wd / "surfaces_1.png"]
    for file_path in file_paths:
        assert file_path.exists()

    assert surfaces.output_files == [str(file_path) for file_path in file_paths]
