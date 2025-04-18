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
"""Test machine learning algorithm selection module."""

from __future__ import annotations

import numpy as np
import pytest

from gemseo.algos.design_space import DesignSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.core.selection import MLAlgoSelection
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.linreg import LinearRegressor
from gemseo.mlearning.regression.algos.polyreg import PolynomialRegressor
from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from gemseo.mlearning.regression.quality.mse_measure import MSEMeasure


@pytest.fixture
def dataset() -> IODataset:
    """The dataset used to train the regression algorithms."""
    data = np.linspace(0, 2 * np.pi, 10)
    data = np.vstack((data, np.sin(data), np.cos(data))).T
    variables = ["x_1", "x_2"]
    variable_names_to_n_components = {"x_1": 1, "x_2": 2}
    variable_names_to_group_names = {
        "x_1": IODataset.INPUT_GROUP,
        "x_2": IODataset.OUTPUT_GROUP,
    }
    return IODataset.from_array(
        data, variables, variable_names_to_n_components, variable_names_to_group_names
    )


def test_init(dataset) -> None:
    """Test construction."""
    selector = MLAlgoSelection(dataset, MSEMeasure)
    assert selector.dataset.equals(dataset)
    assert selector.measure == MSEMeasure
    assert not selector.candidates
    assert not selector.measure_options["multioutput"]


@pytest.mark.parametrize("measure", ["MSEMeasure", MSEMeasure])
def test_init_with_measure(dataset, measure) -> None:
    """Check that the measure can be passed either as a str or a BaseMLAlgoQuality."""
    selector = MLAlgoSelection(dataset, measure)
    assert selector.measure == MSEMeasure


def test_init_fails_if_multioutput_(dataset) -> None:
    expected = (
        "MLAlgoSelection does not support multioutput; "
        "the measure shall return one value."
    )
    with pytest.raises(ValueError, match=expected):
        MLAlgoSelection(dataset, MSEMeasure, multioutput=True)


def test_add_candidate(dataset) -> None:
    """Test add candidate method."""
    selector = MLAlgoSelection(dataset, MSEMeasure)

    # Add linear regression candidate
    selector.add_candidate("LinearRegressor")
    assert selector.candidates
    cand = selector.candidates[0]
    assert isinstance(cand[0], LinearRegressor)
    assert isinstance(cand[1], float)

    # Add polynomial regression candidate
    degrees = [2, 5, 7]
    fit_int = False
    selector.add_candidate(
        "PolynomialRegressor", degree=[2, 5, 7], fit_intercept=[fit_int]
    )
    cand = selector.candidates[-1]
    assert isinstance(cand[0], PolynomialRegressor)
    assert isinstance(cand[1], float)
    assert cand[0]._settings.degree in degrees
    assert cand[0]._settings.fit_intercept == fit_int

    # Add RBF candidate
    space = DesignSpace()
    space.add_variable("smooth", 1, "float", 0.0, 10.0, 0.0)
    algorithm = {"algo_name": "PYDOE_FULLFACT", "n_samples": 11}
    selector.add_candidate("RBFRegressor", space, algorithm)
    cand = selector.candidates[-1]
    assert isinstance(cand[0], RBFRegressor)
    assert isinstance(cand[1], float)
    assert isinstance(cand[0]._settings.smooth, float)
    assert cand[0]._settings.smooth >= 0
    assert cand[0]._settings.smooth <= 10


@pytest.mark.parametrize("measure_evaluation_method_name", ["KFOLDS", "LEARN"])
def test_select(dataset, measure_evaluation_method_name) -> None:
    """Test select method."""
    measure = MSEMeasure
    selector = MLAlgoSelection(
        dataset, measure, measure_evaluation_method_name=measure_evaluation_method_name
    )
    selector.add_candidate("PolynomialRegressor", degree=[1, 2])
    selector.add_candidate("LinearRegressor")
    selector.add_candidate("RBFRegressor", smooth=[0, 0.1, 1, 10])
    algo = selector.select(True)
    assert isinstance(algo, tuple)
    assert len(algo) == 2
    assert isinstance(algo[0], BaseRegressor)
    assert isinstance(algo[1], float)
    cands = selector.candidates
    for cand in cands:
        if cand != algo:
            assert measure.is_better(algo[1], cand[1])
    assert algo[0].__class__.__name__ == "RBFRegressor"

    algo = selector.select()
    assert isinstance(algo, BaseRegressor)
    assert algo.is_trained
