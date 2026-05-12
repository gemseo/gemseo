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

from copy import deepcopy
from pathlib import Path
from unittest import mock

import pytest
from matplotlib import pyplot as plt
from numpy import hstack
from numpy import linspace
from numpy import newaxis

from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.machine_learning.regression.models.linreg import LinearRegressor
from gemseo.post.machine_learning.ml_regressor_quality_viewer import (
    MLRegressorQualityViewer,
)


@pytest.fixture(scope="module")
def linear_regressor() -> LinearRegressor:
    """A linear regressor."""
    x = linspace(0, 1, 10)[:, newaxis]
    y = x**2

    dataset = IODataset()
    dataset.add_input_group(x, "x")
    dataset.add_output_group(hstack((y, y, y)), ["y", "z"], {"y": 2, "z": 1})

    return LinearRegressor(dataset)


@pytest.fixture
def viewer(linear_regressor) -> MLRegressorQualityViewer:
    """The quality viewer for a trained linear regressor."""
    linear_regressor = deepcopy(linear_regressor)
    linear_regressor.learn()
    return MLRegressorQualityViewer(linear_regressor)


def test_residuals(viewer, snapshot_matplotlib) -> None:
    """Check the method plot_residuals_vs_observations."""
    viewer.plot_residuals_vs_observations("y", save=False)


def test_residuals_scatter_no_filter(viewer, snapshot_matplotlib) -> None:
    """Check the method plot_residuals_vs_observations with a list of scatters.

    Do not filter the scatter plots without the residuals.
    """
    viewer.plot_residuals_vs_observations(
        "y", use_scatter_matrix=False, filter_scatters=False, save=False
    )


def test_residuals_scatter(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check the method plot_residuals_vs_observations with a list of scatters."""
    viewer.plot_residuals_vs_observations("y", use_scatter_matrix=False, save=False)


def test_predictions(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check the method plot_predictions_vs_observations."""
    viewer.plot_predictions_vs_observations("y", save=False)


def test_predictions_with_scalar_output(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check the method plot_predictions_vs_observations with a scalar output."""
    viewer.plot_predictions_vs_observations("z", save=False)


def test_predictions_scatter_no_filter(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check the method plot_predictions_vs_observations with a list of scatters.

    Do not filter the scatter plots without the predictions.
    """
    viewer.plot_predictions_vs_observations(
        "y", use_scatter_matrix=False, filter_scatters=False, save=False
    )


def test_predictions_scatter(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check the method plot_predictions_vs_observations with a list of scatters."""
    viewer.plot_predictions_vs_observations("y", use_scatter_matrix=False, save=False)


@pytest.mark.parametrize("input_names", [["x"], "x", ()])
def test_inputs(viewer, input_names, snapshot_matplotlib) -> None:
    """Check the method plot_residuals_vs_inputs."""
    viewer.plot_residuals_vs_inputs("y", input_names, save=False)


def test_inputs_scatter(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check the method plot_residuals_vs_inputs with a list of scatters."""
    viewer.plot_residuals_vs_inputs("y", ["x"], use_scatter_matrix=False, save=False)


def test_input(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check the method plot_residuals_vs_inputs with an input as str."""
    viewer.plot_residuals_vs_inputs("y", "x", save=False)


def test_output(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check the method plot_residuals_vs_observations with output as tuple."""
    viewer.plot_residuals_vs_observations(("y", 0), save=False)


def test_output_scatter(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check the method plot_residuals_vs_observations with a list of scatters."""
    viewer.plot_residuals_vs_observations(
        ("y", 0), use_scatter_matrix=False, save=False
    )


def test_output_scatter_default_file_names(viewer, tmp_wd) -> None:
    """Check the values of the default file names."""
    viewer.plot_residuals_vs_observations(("y", 0))
    assert Path("residuals_vs_observations.png").exists()

    viewer.plot_residuals_vs_observations(("y", 0), use_scatter_matrix=False)
    assert Path("residuals_vs_observations_0.png").exists()
    assert Path("residuals_vs_observations_1.png").exists()


def test_trend(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check plot_residuals_vs_observations with trend."""
    viewer.plot_predictions_vs_observations("y", save=False, trend="rbf")
    assert plt.gcf() == snapshot_matplotlib(tolerance=0.01)


def test_observations(
    viewer,
    snapshot_matplotlib,
) -> None:
    """Check plot_residuals_vs_observations with a validation dataset."""
    observations = IODataset()
    x = linspace(0, 1, 5)[:, newaxis]
    y = x**2
    observations.add_input_group(x, "x")
    observations.add_output_group(hstack((y, y)), "y", {"y": 2})
    viewer.plot_predictions_vs_observations(
        "y",
        observations=observations,
        save=False,
    )


def test_cross_validation_predictions_versus_observations(
    viewer, snapshot_matplotlib
) -> None:
    """Check plot_predictions_vs_observations for a cross validation."""
    viewer.plot_predictions_vs_observations(
        "y",
        observations=MLRegressorQualityViewer.ReferenceDataset.CROSS_VALIDATION,
        save=False,
    )


def test_cross_validation_residuals_versus_observations(
    viewer, snapshot_matplotlib
) -> None:
    """Check plot_residuals_vs_observations for a cross validation."""
    viewer.plot_residuals_vs_observations(
        "y",
        observations=MLRegressorQualityViewer.ReferenceDataset.CROSS_VALIDATION,
        save=False,
    )


def test_cross_validation_residuals_versus_inputs(viewer, snapshot_matplotlib) -> None:
    """Check plot_residuals_vs_inputs for a cross validation."""
    viewer.plot_residuals_vs_inputs(
        "y",
        observations=MLRegressorQualityViewer.ReferenceDataset.CROSS_VALIDATION,
        save=False,
    )


@pytest.mark.parametrize(
    ("model_data", "method_name", "input_names"),
    [
        ("predictions", "plot_predictions_vs_observations", None),
        ("residuals", "plot_residuals_vs_observations", None),
        ("residuals", "plot_residuals_vs_inputs", "input_names"),
    ],
)
def test_signatures(viewer, model_data, method_name, input_names) -> None:
    """Check that the plot methods pass the right values to the core private method."""
    tmp = (input_names,) if input_names else ()
    observations = (
        "observations"
        if method_name != "plot_predictions_vs_observations"
        else MLRegressorQualityViewer.ReferenceDataset.LEARNING
    )
    args = (
        "output",
        *tmp,
        observations,
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
    assert isinstance(method.call_args.kwargs["observations"], Dataset)
    del method.call_args.kwargs["observations"]
    del kwargs["observations"]
    assert method.call_args.kwargs == kwargs
