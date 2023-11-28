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
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from numpy import hstack
from numpy import linspace
from numpy import newaxis

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.post.mlearning.ml_regressor_quality_viewer import MLRegressorQualityViewer
from gemseo.utils.testing.helpers import image_comparison


@pytest.fixture(scope="module")
def viewer() -> MLRegressorQualityViewer:
    """The quality viewer for a linear regressor."""
    x = linspace(0, 1, 10)[:, newaxis]
    y = x**2

    dataset = IODataset()
    dataset.add_input_group(x, "x")
    dataset.add_output_group(hstack((y, y)), "y", {"y": 2})

    model = LinearRegressor(dataset)
    model.learn()

    return MLRegressorQualityViewer(model)


@image_comparison(["residuals"], tol=0.01)
def test_residuals(viewer, pyplot_close_all):
    """Check the method plot_residuals_vs_observations."""
    viewer.plot_residuals_vs_observations("y", save=False, show=False)


@image_comparison(
    [f"residuals_scatter_{chr(x)}" for x in range(ord("a"), ord("a") + 12)], tol=0.01
)
def test_residuals_scatter_no_filter(viewer, pyplot_close_all):
    """Check the method plot_residuals_vs_observations with a list of scatters.

    Do not filter the scatter plots without the residuals.
    """
    viewer.plot_residuals_vs_observations(
        "y",
        use_scatter_matrix=False,
        filter_scatters=False,
        save=False,
        show=False,
    )


@image_comparison(
    [
        f"residuals_scatter_{chr(x)}"
        for x in range(ord("a"), ord("a") + 12)
        if chr(x) not in "il"
    ],
    tol=0.01,
)
def test_residuals_scatter(viewer, pyplot_close_all):
    """Check the method plot_residuals_vs_observations with a list of scatters."""
    viewer.plot_residuals_vs_observations(
        "y", use_scatter_matrix=False, save=False, show=False
    )


@image_comparison(["predictions"], tol=0.01)
def test_predictions(viewer, pyplot_close_all):
    """Check the method plot_predictions_vs_observations."""
    viewer.plot_predictions_vs_observations("y", save=False, show=False)


@image_comparison(
    [f"predictions_scatter_{chr(x)}" for x in range(ord("a"), ord("a") + 12)], tol=0.01
)
def test_predictions_scatter_no_filter(viewer, pyplot_close_all):
    """Check the method plot_predictions_vs_observations with a list of scatters.

    Do not filter the scatter plots without the predictions.
    """
    viewer.plot_predictions_vs_observations(
        "y",
        use_scatter_matrix=False,
        filter_scatters=False,
        save=False,
        show=False,
    )


@image_comparison(
    [
        f"predictions_scatter_{chr(x)}"
        for x in range(ord("a"), ord("a") + 12)
        if chr(x) not in "il"
    ],
    tol=0.01,
)
def test_predictions_scatter(viewer, pyplot_close_all):
    """Check the method plot_predictions_vs_observations with a list of scatters."""
    viewer.plot_predictions_vs_observations(
        "y", use_scatter_matrix=False, save=False, show=False
    )


@pytest.mark.parametrize("input_names", [["x"], "x", ()])
@image_comparison(["inputs"], tol=0.01)
def test_inputs(viewer, pyplot_close_all, input_names):
    """Check the method plot_residuals_vs_inputs."""
    viewer.plot_residuals_vs_inputs("y", input_names, save=False, show=False)


@image_comparison(
    [f"inputs_scatter_{chr(x)}" for x in range(ord("a"), ord("a") + 6)], tol=0.01
)
def test_inputs_scatter(viewer, pyplot_close_all):
    """Check the method plot_residuals_vs_inputs with a list of scatters."""
    viewer.plot_residuals_vs_inputs(
        "y", ["x"], use_scatter_matrix=False, save=False, show=False
    )


@image_comparison(["input"], tol=0.01)
def test_input(viewer, pyplot_close_all):
    """Check the method plot_residuals_vs_inputs with an input as str."""
    viewer.plot_residuals_vs_inputs("y", "x", save=False, show=False)


@image_comparison(["output"], tol=0.01)
def test_output(viewer, pyplot_close_all):
    """Check the method plot_residuals_vs_observations with output as tuple."""
    viewer.plot_residuals_vs_observations(("y", 0), save=False, show=False)


@image_comparison(["output_scatter_a", "output_scatter_b"], tol=0.01)
def test_output_scatter(viewer, pyplot_close_all):
    """Check the method plot_residuals_vs_observations with a list of scatters."""
    viewer.plot_residuals_vs_observations(
        ("y", 0), use_scatter_matrix=False, save=False, show=False
    )


def test_output_scatter_default_file_names(viewer, tmp_wd):
    """Check the values of the default file names."""
    viewer.plot_residuals_vs_observations(("y", 0))
    assert Path("residuals_vs_observations.png").exists()

    viewer.plot_residuals_vs_observations(("y", 0), use_scatter_matrix=False)
    assert Path("residuals_vs_observations_0.png").exists()
    assert Path("residuals_vs_observations_1.png").exists()


@image_comparison(["trend"], tol=0.01)
def test_trend(viewer, pyplot_close_all):
    """Check plot_residuals_vs_observations with trend."""
    viewer.plot_predictions_vs_observations("y", save=False, show=False, trend="rbf")


@image_comparison(["observations"], tol=0.01)
def test_observations(viewer, pyplot_close_all):
    """Check plot_residuals_vs_observations with a validation dataset."""
    observations = IODataset()
    x = linspace(0, 1, 5)[:, newaxis]
    y = x**2
    observations.add_input_group(x, "x")
    observations.add_output_group(hstack((y, y)), "y", {"y": 2})
    viewer.plot_predictions_vs_observations(
        "y", save=False, show=False, observations=observations
    )


@pytest.mark.parametrize(
    ("model_data", "method_name", "input_names"),
    [
        ("predictions", "plot_predictions_vs_observations", None),
        ("residuals", "plot_residuals_vs_observations", None),
        ("residuals", "plot_residuals_vs_inputs", "input_names"),
    ],
)
def test_signatures(viewer, model_data, method_name, input_names):
    """Check that the plot methods pass the right values to the core private method."""
    tmp = (input_names,) if input_names else ()
    args = (
        "output",
        *tmp,
        "observations",
        "use_scatter_matrix",
        "filter_scatters",
    )
    with mock.patch.object(viewer, "_MLRegressorQualityViewer__plot_data") as method:
        getattr(viewer, method_name)(*args, a="a")

    assert method.call_args.args == (
        "output",
        model_data == "residuals",
        method_name[5:],
    )
    kwargs = {
        "a": "a",
        "filter_scatters": "filter_scatters",
        "observations": "observations",
        "use_scatter_matrix": "use_scatter_matrix",
        "save": True,
        "show": False,
    }
    if input_names:
        kwargs["input_names"] = input_names
    assert method.call_args.kwargs == kwargs
