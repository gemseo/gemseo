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
from gemseo.utils.testing.helpers import image_comparison


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
        variable_names_to_n_components={"output": 4},
    )
    dataset.misc["mesh"] = array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    return dataset


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to Surfaces._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "without_option": ({}, ["Surfaces_0", "Surfaces_1"]),
    "with_subsamples": ({"samples": [1]}, ["Surfaces_with_subsamples"]),
    "with_addpoint": (
        {"add_points": True},
        ["Surfaces_with_addpoints_0", "Surfaces_with_addpoints_1"],
    ),
    "with_isolines": (
        {"fill": False},
        ["Surfaces_with_isolines_0", "Surfaces_with_isolines_1"],
    ),
    "with_levels": (
        {"levels": 2},
        ["Surfaces_with_levels_0", "Surfaces_with_levels_1"],
    ),
    "with_properties": (
        {
            "properties": {
                "xlabel": "The xlabel",
                "ylabel": "The ylabel",
                "title": "The title",
            }
        },
        ["Surfaces_properties_0", "Surfaces_properties_1"],
    ),
}


@pytest.mark.parametrize(
    ("kwargs", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot(kwargs, baseline_images, dataset, pyplot_close_all):
    """Test images created by Surfaces._plot against references."""
    properties = kwargs.pop("properties", {})
    plot = Surfaces(dataset, mesh="mesh", variable="output", **kwargs)
    for k, v in properties.items():
        setattr(plot, k, v)
    plot.execute(save=False)


def test_save_multiple_files(dataset, tmp_wd):
    """Check that the generation of multiple files is OK."""
    surfaces = Surfaces(dataset, mesh="mesh", variable="output")
    surfaces.execute()
    file_paths = [tmp_wd / "surfaces_0.png", tmp_wd / "surfaces_1.png"]
    for file_path in file_paths:
        assert file_path.exists()

    assert surfaces.output_files == [str(file_path) for file_path in file_paths]
