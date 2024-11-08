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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from unittest import mock

import pytest

from gemseo import create_discipline
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.uncertainty.sensitivity import correlation_analysis
from gemseo.uncertainty.sensitivity.correlation_analysis import CorrelationAnalysis
from gemseo.utils.compatibility.openturns import IS_OT_LOWER_THAN_1_20
from gemseo.utils.testing.helpers import image_comparison


@pytest.fixture(scope="module")
def correlation() -> CorrelationAnalysis:
    """A correlation analysis."""
    discipline = create_discipline(
        "AnalyticDiscipline", expressions={"y1": "x1+2*x2", "y2": "x1-2*x2"}
    )
    space = ParameterSpace()
    for name in ["x1", "x2"]:
        space.add_random_variable(name, "OTNormalDistribution")
    analysis = CorrelationAnalysis()
    analysis.compute_samples([discipline], space, 100)
    return analysis


def test_indices(correlation) -> None:
    """Check the value indices."""
    indices = correlation.compute_indices()

    # Check that the property indices is the object returned by compute_indices.
    assert indices is correlation.indices

    # Check the methods for which the indices have been computed.
    all_methods = {method.name.lower() for method in correlation.Method}
    available_methods = {field.name for field in fields(indices)}
    if IS_OT_LOWER_THAN_1_20:
        assert available_methods == all_methods - {
            correlation.Method.KENDALL,
            correlation.Method.SSRC,
        }
    else:
        assert available_methods == all_methods

    # Check the names and sizes of the outputs.
    pearson = indices.pearson
    output_names = ["y1", "y2"]
    assert list(pearson) == output_names
    for output_name in output_names:
        assert len(pearson[output_name]) == 1

    # Check the names and sizes of the inputs.
    input_names = ["x1", "x2"]
    assert list(pearson["y1"][0]) == input_names
    for input_name in input_names:
        assert pearson["y1"][0][input_name].shape == (1,)


@pytest.mark.parametrize("baseline_images", [["plot"]])
@pytest.mark.parametrize("output", ["y1", ("y1", 0)])
@image_comparison(None)
def test_plot(correlation, baseline_images, output) -> None:
    """Check the method plot()."""
    correlation.compute_indices()
    correlation.plot(output, save=False, show=False)


@pytest.mark.parametrize("baseline_images", [(["plot_radar"])])
@image_comparison(None)
def test_plot_radar(correlation, baseline_images) -> None:
    """Check the method plot_radar()."""
    correlation.compute_indices()
    correlation.plot_radar("y1", save=False, show=False)


def test_sort_input_variables(correlation) -> None:
    """Check the method sort_input_variables."""
    correlation.compute_indices()
    indices = correlation.indices.spearman["y2"][0]
    c1 = indices["x1"]
    c2 = indices["x2"]

    # The sensitivity index of x1 is negative
    # while the sensitivity index of x2 is positive.
    assert c2 < 0 < c1

    # By taking their absolute values,
    # we can see that the output y2 is more sensitive to x2 than to x1.
    assert abs(c2) > abs(c1)

    # The method sort_input_variables uses
    # the absolute value of the sensitivity indices
    # to sort the input variables from most to least influential.
    assert correlation.sort_input_variables("y2") == ["x2", "x1"]


@pytest.mark.parametrize(
    ("output_names", "expected_keys"),
    [((), ["y1", "y2"]), (("y1",), ["y1"]), (("y2", "y1"), ["y2", "y1"])],
)
def test_main_indices_keys(correlation, output_names, expected_keys) -> None:
    """Check that the keys of the main_indices are the requested output names."""
    correlation.compute_indices(output_names)
    assert list(correlation.main_indices) == expected_keys


def test_mock_ot_version(correlation) -> None:
    """Check that KENDALL and SSRC are not available with openturns < 1.20."""
    correlation.compute_indices()
    assert correlation.indices.kendall
    assert correlation.indices.ssrc
    with mock.patch.object(correlation_analysis, "IS_OT_LOWER_THAN_1_20", new=True):
        correlation.compute_indices()
        assert not correlation.indices.kendall
        assert not correlation.indices.ssrc


def test_from_samples(correlation, tmp_wd):
    """Check the instantiation from samples."""
    file_path = Path("samples.pkl")
    correlation.compute_indices()
    correlation.dataset.to_pickle(file_path)
    new_correlation = CorrelationAnalysis(samples=file_path)
    new_correlation.compute_indices()
    assert new_correlation.indices == correlation.indices
