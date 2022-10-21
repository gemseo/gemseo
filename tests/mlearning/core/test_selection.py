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
from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.selection import MLAlgoSelection
from gemseo.mlearning.qual_measure.mse_measure import MSEMeasure
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.mlearning.regression.polyreg import PolynomialRegressor
from gemseo.mlearning.regression.rbf import RBFRegressor
from gemseo.mlearning.regression.regression import MLRegressionAlgo


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the regression algorithms."""
    data = np.linspace(0, 2 * np.pi, 10)
    data = np.vstack((data, np.sin(data), np.cos(data))).T
    variables = ["x_1", "x_2"]
    sizes = {"x_1": 1, "x_2": 2}
    groups = {"x_1": Dataset.INPUT_GROUP, "x_2": Dataset.OUTPUT_GROUP}
    sample = Dataset()
    sample.set_from_array(data, variables, sizes, groups)
    return sample


def test_init(dataset):
    """Test construction."""
    selector = MLAlgoSelection(dataset, MSEMeasure)
    assert selector.dataset == dataset
    assert selector.measure == MSEMeasure
    assert not selector.candidates
    assert not selector.measure_options["multioutput"]


@pytest.mark.parametrize("measure", ["MSEMeasure", MSEMeasure])
def test_init_with_measure(dataset, measure):
    """Check that the measure can be passed either as a str or a MLQualityMeasure."""
    selector = MLAlgoSelection(dataset, measure)
    assert selector.measure == MSEMeasure


def test_init_fails_if_multioutput_(dataset):
    expected = (
        "MLAlgoSelection does not support multioutput; "
        "the measure shall return one value."
    )
    with pytest.raises(ValueError, match=expected):
        MLAlgoSelection(dataset, MSEMeasure, multioutput=True)


def test_add_candidate(dataset):
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
    assert cand[0].parameters["degree"] in degrees
    assert cand[0].parameters["fit_intercept"] == fit_int

    # Add RBF candidate
    space = DesignSpace()
    space.add_variable("smooth", 1, "float", 0.0, 10.0, 0.0)
    algorithm = {"algo": "fullfact", "n_samples": 11}
    selector.add_candidate("RBFRegressor", space, algorithm)
    cand = selector.candidates[-1]
    assert isinstance(cand[0], RBFRegressor)
    assert isinstance(cand[1], float)
    assert isinstance(cand[0].parameters["smooth"], float)
    assert cand[0].parameters["smooth"] >= 0
    assert cand[0].parameters["smooth"] <= 10


@pytest.mark.parametrize("eval_method", ["kfolds", "learn"])
def test_select(dataset, eval_method):
    """Test select method."""
    measure = MSEMeasure
    selector = MLAlgoSelection(dataset, measure, eval_method=eval_method)
    selector.add_candidate("PolynomialRegressor", degree=[1, 2])
    selector.add_candidate("LinearRegressor")
    selector.add_candidate("RBFRegressor", smooth=[0, 0.1, 1, 10])
    algo = selector.select(True)
    assert isinstance(algo, tuple)
    assert len(algo) == 2
    assert isinstance(algo[0], MLRegressionAlgo)
    assert isinstance(algo[1], float)
    cands = selector.candidates
    for cand in cands:
        if cand != algo:
            assert measure.is_better(algo[1], cand[1])
    assert algo[0].__class__.__name__ == "RBFRegressor"

    algo = selector.select()
    assert isinstance(algo, MLRegressionAlgo)
    assert algo.is_trained
