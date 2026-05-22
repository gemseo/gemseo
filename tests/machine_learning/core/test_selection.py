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
#                         documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test machine learning model selection module."""

from __future__ import annotations

import numpy as np
import pytest

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.datasets.io_dataset import IODataset
from gemseo.machine_learning.core.selection import MLModelSelection
from gemseo.machine_learning.regression.models.base_regressor import BaseRegressor
from gemseo.machine_learning.regression.models.linreg import LinearRegressor
from gemseo.machine_learning.regression.models.linreg_settings import (
    LinearRegressor_Settings,
)
from gemseo.machine_learning.regression.models.polyreg import PolynomialRegressor
from gemseo.machine_learning.regression.models.polyreg_settings import (
    PolynomialRegressor_Settings,
)
from gemseo.machine_learning.regression.models.rbf import RBFRegressor
from gemseo.machine_learning.regression.models.rbf_settings import RBFRegressor_Settings
from gemseo.machine_learning.regression.quality.mse_measure import MSEMeasure
from gemseo.utils.testing.helpers import assert_exception


@pytest.fixture
def dataset() -> IODataset:
    """The dataset used to train the regression models."""
    data = np.linspace(0, 2 * np.pi, 10)
    data = np.vstack((data, np.sin(data), np.cos(data))).T
    variables = ["x_1", "x_2"]
    variable_name_to_n_components = {"x_1": 1, "x_2": 2}
    variable_name_to_group_name = {
        "x_1": IODataset.INPUT_GROUP,
        "x_2": IODataset.OUTPUT_GROUP,
    }
    return IODataset.from_array(
        data, variables, variable_name_to_n_components, variable_name_to_group_name
    )


def test_init(dataset) -> None:
    """Test construction."""
    selector = MLModelSelection(dataset, MSEMeasure)
    assert selector.dataset.equals(dataset)
    assert selector.measure == MSEMeasure
    assert not selector.candidates
    assert not selector.measure_options["multioutput"]


@pytest.mark.parametrize("measure", ["MSEMeasure", MSEMeasure])
def test_init_with_measure(dataset, measure) -> None:
    """Check that the measure can be passed either as a str or a BaseMLModelQuality."""
    selector = MLModelSelection(dataset, measure)
    assert selector.measure == MSEMeasure


def test_init_fails_if_multioutput_(dataset, snapshot) -> None:
    with assert_exception(ValueError, snapshot):
        MLModelSelection(dataset, MSEMeasure, multioutput=True)


def test_add_candidate(dataset) -> None:
    """Test add candidate method."""
    selector = MLModelSelection(dataset, MSEMeasure)

    # Add linear regression candidate
    selector.add_candidate(LinearRegressor_Settings())
    assert selector.candidates
    model, quality = selector.candidates[0]
    assert isinstance(model, LinearRegressor)
    assert quality == pytest.approx(0.40, abs=1e-2)

    # Add polynomial regression candidate with options
    selector.add_candidate(
        PolynomialRegressor_Settings(fit_intercept=False), degree=[2, 5, 7]
    )
    model, quality = selector.candidates[-1]
    assert isinstance(model, PolynomialRegressor)
    assert quality == pytest.approx(0.05, abs=1e-2)
    assert model._settings.degree == 7
    assert model._settings.fit_intercept is False

    # Add RBF candidate with calibration
    space = DesignSpace()
    space.add_variable("smooth", 1, "float", 0.0, 10.0, 0.0)
    selector.add_candidate(
        RBFRegressor_Settings(), space, PYDOE_FULLFACT_Settings(n_samples=1)
    )
    model, quality = selector.candidates[-1]
    assert isinstance(model, RBFRegressor)
    assert quality == pytest.approx(0.12, abs=1e-2)
    assert model._settings.smooth == 5.0

    # Add RBF candidate with calibration and options
    selector.add_candidate(
        RBFRegressor_Settings(),
        space,
        PYDOE_FULLFACT_Settings(n_samples=1),
        epsilon=[0.1, 0.2],
    )
    model, quality = selector.candidates[-1]
    assert isinstance(model, RBFRegressor)
    assert quality == pytest.approx(0.01, abs=1e-2)
    assert model._settings.smooth == 5.0
    assert model._settings.epsilon == 0.1


@pytest.mark.parametrize("measure_evaluation_method_name", ["KFOLDS", "LEARN"])
def test_select(dataset, measure_evaluation_method_name) -> None:
    """Test select method."""
    measure = MSEMeasure
    selector = MLModelSelection(
        dataset, measure, measure_evaluation_method_name=measure_evaluation_method_name
    )
    selector.add_candidate(PolynomialRegressor_Settings(), degree=[1, 2])
    selector.add_candidate(LinearRegressor_Settings())
    selector.add_candidate(RBFRegressor_Settings(), smooth=[0, 0.1, 1, 10])
    model = selector.select(True)
    assert isinstance(model, tuple)
    assert len(model) == 2
    assert isinstance(model[0], BaseRegressor)
    assert isinstance(model[1], float)
    cands = selector.candidates
    for cand in cands:
        if cand != model:
            assert measure.is_better(model[1], cand[1])
    assert model[0].__class__.__name__ == "RBFRegressor"

    model = selector.select()
    assert isinstance(model, BaseRegressor)
    assert model.is_trained
