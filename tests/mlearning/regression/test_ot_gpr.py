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
"""Test the interface to the OpenTURNS' Kriging."""

from __future__ import annotations

from unittest import mock
from unittest.mock import Mock

import openturns
import pytest
from numpy import array
from numpy import hstack
from numpy import ndarray
from numpy import zeros
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from openturns import CovarianceMatrix
from openturns import GeneralizedExponential
from openturns import KrigingAlgorithm
from openturns import KrigingResult
from openturns import MaternModel
from openturns import NLopt
from openturns import ResourceMap
from packaging import version
from scipy.optimize import rosen

from gemseo import execute_algo
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.algos.ot_gpr import OTGaussianProcessRegressor
from gemseo.mlearning.regression.algos.ot_gpr_settings import CovarianceModel
from gemseo.mlearning.regression.algos.ot_gpr_settings import Trend
from gemseo.problems.optimization.rosenbrock import Rosenbrock

OTGaussianProcessRegressor.HMATRIX_ASSEMBLY_EPSILON = 1e-10
OTGaussianProcessRegressor.HMATRIX_RECOMPRESSION_EPSILON = 1e-10
# The EPSILONs are reduced to make the HMAT-based Kriging interpolating.

OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK = 9


# The maximum learning sample size to use LAPACK is reduced to accelerate the tests.


def func(x: ndarray) -> float:
    """Return the sum of the components of a vector.

    Args:
        x: A vector.

    Returns:
        The sum of the components of the vector.
    """
    return sum(x)


@pytest.fixture(scope="module")
def problem() -> Rosenbrock:
    """The Rosenbrock problem with an observable summing the inputs."""
    rosenbrock = Rosenbrock()
    rosenbrock.add_observable(MDOFunction(func, "sum"))
    return rosenbrock


@pytest.fixture(scope="module")
def dataset(problem) -> IODataset:
    """A 9-length full-factorial sampling of the Rosenbrock problem."""
    execute_algo(problem, algo_name="PYDOE_FULLFACT", n_samples=9, algo_type="doe")
    return problem.to_dataset(opt_naming=False)


@pytest.fixture(scope="module")
def dataset_2(problem) -> IODataset:
    """A 9-length full-factorial sampling of the Rosenbrock problem."""
    execute_algo(problem, algo_name="PYDOE_FULLFACT", n_samples=9, algo_type="doe")
    data = problem.to_dataset(opt_naming=False)
    data.add_variable(
        "rosen2",
        hstack((
            data.get_view(variable_names="rosen").to_numpy(),
            -data.get_view(variable_names="rosen").to_numpy(),
        )),
        group_name=data.OUTPUT_GROUP,
    )
    return data


@pytest.fixture(scope="module")
def kriging(dataset) -> OTGaussianProcessRegressor:
    """A Kriging model trained on the Rosenbrock dataset."""
    model = OTGaussianProcessRegressor(dataset)
    model.learn()
    return model


def test_class_constants(kriging):
    """Check the class constants."""
    assert kriging.LIBRARY == "OpenTURNS"
    assert kriging.SHORT_ALGO_NAME == "GPR"


@pytest.mark.parametrize(
    ("n_samples", "use_hmat"),
    [
        (1, False),
        (OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK, False),
        (OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK + 1, True),
    ],
)
def test_kriging_use_hmat_default(n_samples, use_hmat):
    """Check the default library (LAPACK or HMAT) according to the sample size."""
    dataset = IODataset.from_array(
        zeros((n_samples, 2)),
        variable_names=["in", "out"],
        variable_names_to_group_names={
            "in": IODataset.INPUT_GROUP,
            "out": IODataset.OUTPUT_GROUP,
        },
    )
    assert OTGaussianProcessRegressor(dataset).use_hmat is use_hmat


@pytest.mark.parametrize("use_hmat", [True, False])
def test_kriging_use_hmat(dataset, use_hmat):
    """Check that the HMAT can be specified at initialization or after."""
    kriging = OTGaussianProcessRegressor(dataset, use_hmat=use_hmat)
    # Check at initialization
    assert kriging.use_hmat is use_hmat

    # Check after initialization
    kriging.use_hmat = not use_hmat
    assert kriging.use_hmat is not use_hmat


def test_kriging_predict_on_learning_set(dataset):
    """Check that the Kriging interpolates the learning set."""
    kriging = OTGaussianProcessRegressor(dataset)
    kriging.learn()
    for x in kriging.learning_set.get_view(
        group_names=IODataset.INPUT_GROUP
    ).to_numpy():
        prediction = kriging.predict({"x": x})
        assert_allclose(prediction["sum"], sum(x), atol=1e-3)
        assert_allclose(prediction["rosen"], rosen(x))


@pytest.mark.parametrize("x1", [-1, 1])
@pytest.mark.parametrize("x2", [-1, 1])
def test_kriging_predict(dataset, x1, x2):
    """Check that the Kriging is not yet good enough to extrapolate."""
    kriging = OTGaussianProcessRegressor(dataset, multi_start_n_samples=1)
    kriging.learn()
    x = array([x1, x2])
    prediction = kriging.predict({"x": x})
    assert prediction["sum"] != pytest.approx(sum(x))
    assert prediction["rosen"] != pytest.approx(rosen(x))


@pytest.mark.parametrize("transformer", [{}, {"inputs": "MinMaxScaler"}])
def test_kriging_predict_std_on_learning_set(transformer, dataset):
    """Check that the standard deviation is correctly predicted for a learning point.

    The standard deviation should be equal to zero.
    """
    kriging = OTGaussianProcessRegressor(dataset, transformer=transformer)
    kriging.learn()
    for x in kriging.learning_set.get_view(
        group_names=IODataset.INPUT_GROUP
    ).to_numpy():
        assert_allclose(kriging.predict_std(x), 0, atol=1e-1)


@pytest.mark.parametrize("x1", [-1, 1])
@pytest.mark.parametrize("x2", [-1, 1])
@pytest.mark.parametrize("transformer", [{}, {"inputs": "MinMaxScaler"}])
def test_kriging_predict_std(transformer, dataset, x1, x2):
    """Check that the standard deviation is correctly predicted for a validation point.

    The standard deviation should be the square root of the variance computed by the
    method KrigingResult.getConditionalCovariance of OpenTURNS.
    """
    kriging = OTGaussianProcessRegressor(dataset, transformer=transformer)
    original_method = KrigingResult.getConditionalCovariance
    v1 = 4.0 + x1 + x2
    v2 = 9.0 + x1 + x2
    KrigingResult.getConditionalCovariance = Mock(
        return_value=CovarianceMatrix(2, [v1, 0.5, 0.5, v2])
    )
    kriging.learn()
    assert_allclose(kriging.predict_std(array([x1, x2])), array([v1, v2]) ** 0.5)
    KrigingResult.getConditionalCovariance = original_method


def test_kriging_predict_jacobian(kriging):
    """Check the shape of the Jacobian."""
    jacobian = kriging.predict_jacobian(array([[0.0, 0.0], [-2.0, -2.0], [2.0, 2.0]]))
    assert jacobian.shape == (3, 2, 2)


@pytest.mark.parametrize("output_name", ["rosen", "rosen2"])
@pytest.mark.parametrize(
    "input_data",
    [
        array([1.0, 1.0]),
        array([[1.0, 1.0]]),
        {"x": array([1.0, 1.0])},
        {"x": array([[1.0, 1.0]])},
    ],
)
def test_kriging_std_output_dimension(dataset_2, output_name, input_data):
    """Check the shape of the array returned by predict_std()."""
    model = OTGaussianProcessRegressor(dataset_2, output_names=[output_name])
    model.learn()
    ndim = model.output_dimension
    if isinstance(input_data, dict):
        one_sample = input_data["x"].ndim == 1
    else:
        one_sample = input_data.ndim == 1
    shape = (ndim,) if one_sample else (1, ndim)

    assert model.predict_std(input_data).shape == shape


@pytest.mark.parametrize(
    ("trend", "shape"),
    [
        (Trend.CONSTANT, (2, 1)),
        (Trend.LINEAR, (2, 3)),
        (Trend.QUADRATIC, (2, 6)),
    ],
)
def test_trend_type(dataset, trend, shape):
    """Check the trend type of the Gaussian process regressor."""
    model = OTGaussianProcessRegressor(dataset, trend=trend)
    model.learn()
    if version.parse(openturns.__version__) < version.parse("1.21"):
        assert array(model.algo.getTrendCoefficients()).shape == shape
    else:
        assert array(model.algo.getTrendCoefficients()).shape == (shape[0] * shape[1],)


def test_default_optimizer(dataset):
    """Check that the default optimizer is TNC."""
    model = OTGaussianProcessRegressor(dataset, multi_start_n_samples=1)
    with mock.patch.object(KrigingAlgorithm, "setOptimizationAlgorithm") as method:
        model.learn()

    assert method.call_args.args[0].__class__.__name__ == "TNC"


def test_custom_optimizer(dataset):
    """Check that the optimizer can be changed."""
    optimizer = NLopt("LN_NELDERMEAD")
    model = OTGaussianProcessRegressor(
        dataset, optimizer=optimizer, multi_start_n_samples=0
    )
    with mock.patch.object(KrigingAlgorithm, "setOptimizationAlgorithm") as method:
        model.learn()

    assert method.call_args.args[0] == optimizer


@pytest.mark.parametrize(
    "optimization_space_type",
    [None, "empty", "full"],
)
def test_custom_optimization_space(dataset, optimization_space_type):
    """Check that the optimization bounds can be changed."""
    template = "GeneralLinearModelAlgorithm-DefaultOptimization{}Bound"
    ResourceMap.GetAsScalar(template.format("Upper"))
    lower_bound = [ResourceMap.GetAsScalar(template.format("Lower"))] * 4
    upper_bound = [ResourceMap.GetAsScalar(template.format("Upper"))] * 4

    optimization_space = DesignSpace()
    if optimization_space_type is None:
        optimization_space = None
    elif optimization_space_type == "full":
        lower_bound = [1e-2, 1e-3, 1e-2, 1e-3]
        upper_bound = [1e1, 1e2, 1e1, 1e2]
        optimization_space.add_variable(
            "x",
            size=4,
            lower_bound=array(lower_bound),
            upper_bound=array(upper_bound),
        )

    model = OTGaussianProcessRegressor(
        dataset, optimization_space=optimization_space, multi_start_n_samples=0
    )
    with mock.patch.object(KrigingAlgorithm, "setOptimizationBounds") as method:
        model.learn()

    interval = method.call_args.args[0]
    assert interval.getLowerBound() == lower_bound
    assert interval.getUpperBound() == upper_bound


def test_multi_start_optimization(dataset):
    """Check that the multi-start optimization works."""
    model = OTGaussianProcessRegressor(dataset)
    model.learn()
    reference_length_scales = model.algo.getCovarianceModel().getFullParameter()
    model.algo.getTrendCoefficients()

    # Here we check that the optimization process is deterministic.
    model = OTGaussianProcessRegressor(dataset)
    model.learn()
    assert model.algo.getCovarianceModel().getFullParameter() == reference_length_scales

    # Here we check that multi-start optimization leads to a different solution.
    model = OTGaussianProcessRegressor(dataset, multi_start_n_samples=2)
    model.learn()
    assert model.algo.getCovarianceModel().getFullParameter() != reference_length_scales

    # Here we check that the multi_start_xxx arguments are passed to OpenTURNS.
    model = OTGaussianProcessRegressor(
        dataset,
        multi_start_algo_name="LHS",
        multi_start_n_samples=9,
        multi_start_algo_settings={"strength": 2},
    )
    with mock.patch.object(KrigingAlgorithm, "setOptimizationAlgorithm") as method:
        model.learn()

    optimizer = method.call_args.args[0]
    assert optimizer.__class__.__name__ == "MultiStart"
    doe = array(optimizer.getStartingSample())
    ot_interval = model._OTGaussianProcessRegressor__optimization_space
    design_space = DesignSpace()
    design_space.add_variable(
        "x",
        ot_interval.getDimension(),
        lower_bound=ot_interval.getLowerBound(),
        upper_bound=ot_interval.getUpperBound(),
    )
    doe_algo = DOELibraryFactory().create("LHS")
    assert_equal(doe, doe_algo.compute_doe(design_space, n_samples=9, strength=2))


def test_default_covariance_model(dataset):
    """Check default covariance model is SquaredExponential."""
    model = OTGaussianProcessRegressor(dataset)
    model.learn()
    assert "Matern" in str(model.algo.getCovarianceModel())
    assert "nu=2.5" in str(model.algo.getCovarianceModel())


@pytest.mark.parametrize(
    ("covariance_model", "nu"),
    [
        (MaternModel, 1.5),
        (MaternModel(2), 1.5),
        ([MaternModel, GeneralizedExponential], 1.5),
        ([MaternModel(2), GeneralizedExponential], 1.5),
        ([MaternModel, GeneralizedExponential(2)], 1.5),
        ([MaternModel(2), GeneralizedExponential(2)], 1.5),
        (CovarianceModel.MATERN12, 0.5),
        (
            [
                CovarianceModel.MATERN12,
                GeneralizedExponential(2),
            ],
            0.5,
        ),
    ],
)
def test_custom_covariance_model(dataset, covariance_model, nu):
    """Check that the covariance model can be changed."""
    use_other_covariance_model = isinstance(covariance_model, list)
    model = OTGaussianProcessRegressor(dataset, covariance_model=covariance_model)
    model.learn()
    covariance_model_str = str(model.algo.getCovarianceModel())
    assert "MaternModel" in covariance_model_str
    assert f"nu={nu}" in covariance_model_str
    if use_other_covariance_model:
        assert "GeneralizedExponential" in covariance_model_str


@pytest.mark.parametrize("kernel_type", CovarianceModel)
def test_covariance_kernel_type(dataset, kernel_type):
    """Check the attribute CovarianceModel."""
    model = OTGaussianProcessRegressor(dataset, covariance_model=kernel_type)
    model.learn()
    name = kernel_type.value
    covariance_model_str = str(model.algo.getCovarianceModel())
    if name.startswith("Matern"):
        assert "Matern" in covariance_model_str
        assert f"nu={name[6]}.{name[7]})"
    else:
        assert name in covariance_model_str


@pytest.mark.parametrize(
    ("compute_options", "init_options", "expected_samples", "expected_samples_1pt"),
    [
        (
            {},
            {},
            array([
                [
                    [1.47647972e03, 2.21396378e00],
                    [1.20973761e03, 1.18612131e00],
                    [1.25126291e03, 2.18171156e00],
                ],
                [
                    [1.49048507e03, -1.51470882e00],
                    [1.55715040e03, 1.20897280e00],
                    [1.49389310e03, -3.14649015e00],
                ],
            ]),
            array([[1.39052263e03, 1.44782222e-01], [1.24957800e03, -2.77257687e00]]),
        ),
        (
            {"seed": 2},
            {},
            array([
                [
                    [1.39052263e03, 1.44782222e-01],
                    [1.25298244e03, -2.72898984e00],
                    [1.41856634e03, 4.59435414e-01],
                ],
                [
                    [1.53954913e03, 4.42708465e00],
                    [1.45285976e03, 6.24822690e-01],
                    [1.50965565e03, -3.20010056e00],
                ],
            ]),
            array([[1.39052263e03, 1.44782222e-01], [1.24957800e03, -2.77257687e00]]),
        ),
        (
            {},
            {"output_names": ["sum"]},
            array([
                [[0.42115111], [1.42419408], [1.02099548]],
                [[0.420637], [1.42092809], [1.02177389]],
            ]),
            array([[0.42043643], [0.42034654]]),
        ),
    ],
)
@pytest.mark.parametrize("transformer", [{}, {"inputs": "Scaler", "outputs": "Scaler"}])
def test_compute_samples(
    dataset,
    compute_options,
    init_options,
    expected_samples,
    expected_samples_1pt,
    transformer,
):
    """Check the method compute_samples."""
    model = OTGaussianProcessRegressor(dataset, transformer=transformer, **init_options)
    model.learn()
    input_data = array([[0.23, 0.19], [0.73, 0.69], [0.13, 0.89]])
    samples = model.compute_samples(input_data, 2, **compute_options)
    assert_allclose(samples, expected_samples, rtol=1e-4)

    samples = model.compute_samples(input_data[0], 2, **compute_options)
    assert_allclose(samples, expected_samples_1pt, rtol=1e-4)
