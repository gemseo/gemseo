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
"""Test machine learning algorithm calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import array_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.mlearning.core.calibration import MLAlgoAssessor
from gemseo.mlearning.core.calibration import MLAlgoCalibration
from gemseo.mlearning.quality_measures.mse_measure import MSEMeasure
from gemseo.problems.dataset.rosenbrock import create_rosenbrock_dataset

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """The dataset used to train the regression algorithms."""
    return create_rosenbrock_dataset(opt_naming=False)


def test_discipline_multioutput_fail(dataset):
    """Verify that MLAlgoAssessor raises an error if multioutput option is True."""
    with pytest.raises(
        ValueError,
        match="MLAlgoAssessor does not support multioutput.",
    ):
        MLAlgoAssessor(
            "PolynomialRegressor",
            dataset,
            ["degree"],
            MSEMeasure,
            measure_evaluation_method_name="LOO",
            measure_options={"multioutput": True},
        )


@pytest.mark.parametrize("options", [{"multioutput": False}, {}])
def test_discipline_multioutput(dataset, options):
    """Verify that MLAlgoAssessor works correctly when multioutput option is False."""
    assessor = MLAlgoAssessor(
        "PolynomialRegressor",
        dataset,
        ["degree"],
        MSEMeasure,
        measure_evaluation_method_name="LOO",
        measure_options=options,
    )
    assert not assessor.measure_options["multioutput"]


def test_discipline(dataset):
    """Test discipline."""
    disc = MLAlgoAssessor(
        "PolynomialRegressor",
        dataset,
        ["degree"],
        MSEMeasure,
        measure_evaluation_method_name="LOO",
    )
    result = disc.execute({"degree": array([3])})
    assert "degree" in result
    assert "criterion" in result
    assert "learning" in result
    assert allclose(result["criterion"], array([32107]), atol=1e0)
    assert array_equal(result["degree"], array([3]))


@pytest.fixture(scope="module")
def calibration_space() -> DesignSpace:
    """The space of the parameters to be calibrated."""
    calibration_space = DesignSpace()
    calibration_space.add_variable("penalty_level", 1, "float", 0.0, 1.0, 0.5)
    return calibration_space


@pytest.mark.parametrize(
    "algo", [("fullfact", "n_samples"), ("NLOPT_COBYLA", "max_iter")]
)
def test_calibration(dataset, calibration_space, algo):
    """Test calibration."""
    n_samples = 2
    calibration = MLAlgoCalibration(
        "PolynomialRegressor",
        dataset,
        ["penalty_level"],
        calibration_space,
        MSEMeasure,
        measure_evaluation_method_name="LOO",
        degree=2,
    )

    assert calibration.get_history("learning") is None

    calibration.execute({"algo": algo[0], algo[1]: n_samples})
    x_opt = calibration.optimal_parameters
    f_opt = calibration.optimal_criterion
    algo_opt = calibration.optimal_algorithm

    assert algo_opt.parameters["penalty_level"] == x_opt["penalty_level"]
    assert calibration.get_history("penalty_level").shape == (n_samples, 1)
    assert calibration.get_history("criterion").shape == (n_samples, 1)
    assert calibration.get_history("learning").shape == (n_samples, 1)
    assert len(calibration.algos) == n_samples

    calibration.maximize_objective = True
    calibration.execute({"algo": algo[0], algo[1]: n_samples})
    assert -calibration.optimal_criterion > f_opt
