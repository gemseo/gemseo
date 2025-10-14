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

import re

import pytest
from numpy import array
from numpy import hstack
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal

from gemseo import sample_disciplines
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.linear_model_fitting.omp_settings import (
    OrthogonalMatchingPursuit_Settings,
)
from gemseo.mlearning.linear_model_fitting.ridge_settings import Ridge_Settings
from gemseo.mlearning.regression.algos.fce import FCERegressor
from gemseo.mlearning.regression.algos.fce_settings import FCERegressor_Settings
from gemseo.mlearning.regression.algos.fce_settings import OrthonormalFunctionBasis
from gemseo.mlearning.regression.quality.r2_measure import R2Measure
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.problems.uncertainty.utils import UniformDistribution
from gemseo.utils.comparisons import compare_dict_of_arrays


@pytest.fixture(scope="module", params=[False, True])
def multioutput(request) -> bool:
    """The problem is multioutput."""
    return request.param


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """The Ishigami discipline with a second output."""
    d = AnalyticDiscipline({
        "y": "sin(x1)*(1+0.1*x3**4)+7*sin(x2)**2",
        "z": "y+a",
    })
    d.io.input_grammar.defaults["a"] = array([1.0])
    return d


@pytest.fixture(scope="module")
def uncertain_space() -> IshigamiSpace:
    """The Ishigami uncertain space."""
    return IshigamiSpace(uniform_distribution_name=UniformDistribution.OPENTURNS)


@pytest.fixture(scope="module")
def dataset(multioutput, discipline, uncertain_space) -> IODataset:
    """An Ishigami training dataset containing special Jacobian data.."""
    return sample_disciplines(
        [discipline],
        uncertain_space,
        ["y", "z"] if multioutput else "y",
        algo_settings_model=OT_OPT_LHS_Settings(n_samples=100, eval_jac=True),
        formulation_settings={"differentiated_input_names_substitute": ("a",)},
    )


@pytest.fixture(scope="module")
def dataset2(multioutput, discipline, uncertain_space) -> IODataset:
    """An Ishigami training dataset containing Jacobian data."""
    return sample_disciplines(
        [discipline],
        uncertain_space,
        ["y", "z"] if multioutput else "y",
        algo_settings_model=OT_OPT_LHS_Settings(n_samples=50, eval_jac=True),
    )


@pytest.fixture(scope="module")
def validation_dataset(multioutput, discipline, uncertain_space) -> IODataset:
    """An Ishigami validation dataset."""
    return sample_disciplines(
        [discipline],
        uncertain_space,
        ["y", "z"] if multioutput else "y",
        algo_settings_model=OT_OPT_LHS_Settings(n_samples=1000),
    )


@pytest.fixture(scope="module")
def regressor(dataset) -> FCERegressor:
    """An FCE."""
    regressor_ = FCERegressor(
        dataset, FCERegressor_Settings(use_special_jacobian_data=True)
    )
    regressor_.learn()
    return regressor_


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("mean", [3.531478, 4.531478]),
        ("variance", [3.582373, 3.582373]),
        ("standard_deviation", [3.582373**0.5, 3.582373**0.5]),
    ],
)
def test_statistics(regressor, multioutput, name, value):
    """Check the value of the statistics."""
    assert_allclose(
        getattr(regressor, name),
        array(value) if multioutput else array([value[0]]),
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("input_data", "output_data"),
    [
        (array([1.0, 1.0, 1.0]), array([5.139477])),
        (array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), array([[5.139477], [6.194871]])),
    ],
)
def test_predict(regressor, input_data, output_data, multioutput):
    """Check whether the predict method."""
    if multioutput:
        output_data = hstack((output_data, output_data + 1))
    assert_allclose(regressor.predict(input_data), output_data, rtol=1e-5)


@pytest.mark.parametrize(
    ("input_data", "jacobian_data"),
    [
        (array([1.0, 1.0, 1.0]), array([[1.048738, -0.25861, 0.206839]])),
        (
            array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            array([
                [[1.048738, -0.25861, 0.206839]],
                [[1.184464, -0.572983, 0.502342]],
            ]),
        ),
    ],
)
def test_predict_jacobian(regressor, input_data, jacobian_data, multioutput):
    """Check whether the predict_jacobian method."""
    if multioutput:
        if input_data.ndim == 1:
            jacobian_data = array([
                [1.048738, -0.25861, 0.206839],
                [1.048738, -0.25861, 0.206839],
            ])
        else:
            jacobian_data = array([
                [
                    [1.048738, -0.25861, 0.206839],
                    [1.048738, -0.25861, 0.206839],
                ],
                [
                    [1.18446438, -0.57298323, 0.50234162],
                    [1.18446438, -0.57298323, 0.50234162],
                ],
            ])

    assert_allclose(regressor.predict_jacobian(input_data), jacobian_data, atol=1e-6)


def test_first_sobol_indices(regressor, multioutput):
    """Check the first_sobol_indices property."""

    first_sobol_indices = regressor.first_sobol_indices
    assert len(first_sobol_indices) == 1 + bool(multioutput)
    expected_indices = (
        (
            {
                "x1": array([0.76795552]),
                "x2": array([0.08659672]),
                "x3": array([0.00771764]),
            },
            {
                "x1": array([0.76795552]),
                "x2": array([0.08659672]),
                "x3": array([0.00771764]),
            },
        )
        if multioutput
        else (
            {
                "x1": array([0.76795552]),
                "x2": array([0.08659672]),
                "x3": array([0.00771764]),
            },
        )
    )
    for i, indices in enumerate(first_sobol_indices):
        assert compare_dict_of_arrays(
            indices,
            expected_indices[i],
            tolerance=1e-6,
        )


def test_total_order_sobol_indices(regressor, multioutput):
    """Check the total_sobol_indices property."""
    total_sobol_indices = regressor.total_sobol_indices
    assert len(total_sobol_indices) == 1 + bool(multioutput)
    expected_indices = (
        (
            {
                "x1": array([0.893784]),
                "x2": array([0.09856986]),
                "x3": array([0.14537625]),
            },
            {
                "x1": array([0.893784]),
                "x2": array([0.09856986]),
                "x3": array([0.14537625]),
            },
        )
        if multioutput
        else (
            {
                "x1": array([0.893784]),
                "x2": array([0.09856986]),
                "x3": array([0.14537625]),
            },
        )
    )
    for i, indices in enumerate(total_sobol_indices):
        assert compare_dict_of_arrays(
            indices,
            expected_indices[i],
            tolerance=1e-6,
        )


def test_differentiation_wrt_special_variables(regressor, multioutput):
    """Check that a FCE can be differentiated with respect to special variables."""
    jacobian = regressor.predict_jacobian_wrt_special_variables(array([1.0, 2.0, 3.0]))
    assert_almost_equal(
        jacobian, array([[0.0], [1.0]]) if multioutput else array([[0.0]])
    )
    assert_almost_equal(
        regressor.mean_jacobian_wrt_special_variables,
        array([[0.0], [1.0]]) if multioutput else array([[0.0]]),
    )
    assert_almost_equal(
        regressor.variance_jacobian_wrt_special_variables,
        array([[0.0], [0.0]]) if multioutput else array([[0.0]]),
    )
    assert_almost_equal(
        regressor.standard_deviation_jacobian_wrt_special_variables,
        array([[0.0], [0.0]]) if multioutput else array([[0.0]]),
    )


def test_settings():
    """Check FCERegressor_Settings."""
    assert isinstance(
        FCERegressor_Settings().linear_model_fitter_settings,
        OrthogonalMatchingPursuit_Settings,
    )
    ridge_settings = Ridge_Settings()
    assert (
        FCERegressor_Settings(
            linear_model_fitter_settings=ridge_settings
        ).linear_model_fitter_settings
        is ridge_settings
    )


@pytest.mark.parametrize(
    ("basis", "mean", "mean2", "variance", "variance2"),
    [
        (
            OrthonormalFunctionBasis.POLYNOMIAL,
            array([3.5314783]),
            array([3.5314783, 4.5314783]),
            array([3.5823731]),
            array([3.5823731, 3.5823731]),
        ),
        (
            OrthonormalFunctionBasis.FOURIER,
            array([3.5339085]),
            array([3.5339085, 4.5339085]),
            array([4.6279828]),
            array([4.6279828, 4.6279828]),
        ),
        (
            OrthonormalFunctionBasis.HAAR,
            array([3.5421638]),
            array([3.5421638, 4.5421638]),
            array([4.2673207]),
            array([4.2673207, 4.2673207]),
        ),
    ],
)
def test_basis(dataset, basis, mean, mean2, variance, variance2, multioutput):
    """Check different orthonormal function bases."""
    settings = FCERegressor_Settings(basis=basis)
    fce = FCERegressor(dataset, settings_model=settings)
    fce.learn()
    assert_almost_equal(fce.mean, mean2 if multioutput else mean)
    assert_almost_equal(fce.variance, variance2 if multioutput else variance)


def test_gradient_enhanced_fce(dataset2, validation_dataset):
    """Check that gradient-enhanced FCE is better than FCE."""
    fce = FCERegressor(dataset2, FCERegressor_Settings(degree=4))
    fce.learn()

    ge_fce = FCERegressor(
        dataset2, FCERegressor_Settings(degree=4, learn_jacobian_data=True)
    )
    ge_fce.learn()

    r2 = R2Measure(fce).compute_test_measure(validation_dataset)
    r2_ge = R2Measure(ge_fce).compute_test_measure(validation_dataset)
    assert all(r2_ge > r2)


def test_fce_settings_jacobian_error():
    """Check the error raised when combining Jacobian data and special Jacobian data."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Only one of the options learn_jacobian_data and use_special_jacobian_data "
            "can be True."
        ),
    ):
        FCERegressor_Settings(use_special_jacobian_data=True, learn_jacobian_data=True)


@pytest.mark.parametrize(
    "option_name", ["learn_jacobian_data", "use_special_jacobian_data"]
)
def test_fce_jacobian_error(option_name):
    """Check the error when enabling learn_jacobian_data without Jacobian data."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Option {option_name} is True "
            "but the training dataset does not contain Jacobian data."
        ),
    ):
        FCERegressor(IODataset(), FCERegressor_Settings(**{option_name: True}))


def test_basis_functions(regressor):
    """Check the basis_functions attribute."""
    basis_functions = regressor._basis_functions
    assert len(basis_functions) == 10
    result = array(basis_functions[1](array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])))
    assert_almost_equal(result, array([[1.7320508], [3.4641016]]))


def test_isoprobabilistic_transformation(regressor):
    """Check the isoprobabilistic_transformation attribute."""
    result = regressor._isoprobabilistic_transformation(
        array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    )
    assert_almost_equal(
        result,
        array([[0.3183099, 0.3183099, 0.3183099], [0.6366198, 0.6366198, 0.6366198]]),
    )
