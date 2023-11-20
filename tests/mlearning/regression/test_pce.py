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
"""Test polynomial chaos expansion regression module."""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from pickle import dump
from pickle import load

import pytest
from numpy import array
from numpy import pi
from numpy import sin
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from openturns import FunctionalChaosRandomVector

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.doe_scenario import DOEScenario
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.mlearning import import_regression_model
from gemseo.mlearning.quality_measures.r2_measure import R2Measure
from gemseo.mlearning.regression.pce import CleaningOptions
from gemseo.mlearning.regression.pce import PCERegressor
from gemseo.utils.comparisons import compare_dict_of_arrays


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """A linear discipline with two outputs."""
    return AnalyticDiscipline({"y1": "1+2*x1+3*x2", "y2": "-1-2*x1-3*x2"})


@pytest.fixture(scope="module")
def probability_space() -> ParameterSpace:
    """The probability space associated with the linear discipline."""
    space = ParameterSpace()
    space.add_random_variable("x1", "OTUniformDistribution")
    space.add_random_variable("x2", "OTUniformDistribution")
    return space


@pytest.fixture(scope="module")
def dataset(discipline, probability_space) -> IODataset:
    """The learning dataset associated with the linear discipline."""
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y1", probability_space)
    scenario.add_observable("y2")
    scenario.execute({"algo": "fullfact", "n_samples": 9})
    dataset = scenario.to_dataset(opt_naming=False)
    dataset.add_variable("weight", 1)
    return dataset


@pytest.fixture(scope="module")
def ishigami_discipline() -> AnalyticDiscipline:
    """The Ishigami discipline."""
    return AnalyticDiscipline({"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"})


@pytest.fixture(scope="module")
def ishigami_probability_space() -> ParameterSpace:
    """The probability space associated with the Ishigami discipline."""
    space = ParameterSpace()
    space.add_random_variable("x1", "OTUniformDistribution", minimum=-pi, maximum=pi)
    space.add_random_variable("x2", "OTUniformDistribution", minimum=-pi, maximum=pi)
    space.add_random_variable("x3", "OTUniformDistribution", minimum=-pi, maximum=pi)
    return space


@pytest.fixture(scope="module")
def ishigami_dataset(ishigami_discipline, ishigami_probability_space) -> IODataset:
    """The learning dataset associated with the Ishigami discipline."""
    scenario = DOEScenario(
        [ishigami_discipline], "DisciplinaryOpt", "y", ishigami_probability_space
    )
    scenario.execute({"algo": "fullfact", "n_samples": 125})
    return scenario.to_dataset(opt_naming=False)


@pytest.fixture(scope="module")
def pce(dataset, probability_space) -> PCERegressor:
    """A PCERegressor trained from the dataset associated with the linear discipline."""
    model = PCERegressor(dataset, probability_space)
    model.learn()
    return model


@pytest.fixture(scope="module")
def quadrature_points(discipline, probability_space) -> IODataset:
    """The quadrature points computed by a PCERegressor with degree equal to 1."""
    model = PCERegressor(
        None, probability_space, use_quadrature=True, discipline=discipline
    )
    return model.learning_set


@pytest.fixture(scope="module")
def untrained_pce(dataset, probability_space) -> PCERegressor:
    """An untrained PCERegressor for the linear discipline."""
    return PCERegressor(dataset, probability_space)


def test_discipline_and_data_with_quadrature(dataset, discipline, probability_space):
    """Check that quadrature cannot be used with both a dataset and a discipline."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The quadrature rule requires data or discipline but not both."
        ),
    ):
        PCERegressor(
            dataset, probability_space, discipline=discipline, use_quadrature=True
        )


def test_no_discipline_and_no_data_with_quadrature(probability_space):
    """Check that quadrature requires either a dataset or a discipline."""
    with pytest.raises(
        ValueError,
        match=re.escape("The quadrature rule requires either data or discipline."),
    ):
        PCERegressor(None, probability_space, discipline=None, use_quadrature=True)


def test_lars_with_quadrature(discipline, probability_space):
    """Check that LARS is not applicable with quadrature."""
    with pytest.raises(
        ValueError,
        match=re.escape("LARS is not applicable with the quadrature rule."),
    ):
        PCERegressor(
            None,
            probability_space,
            discipline=discipline,
            use_quadrature=True,
            use_lars=True,
        )


def test_no_dataset_with_least_square(probability_space, discipline):
    """Check that least square requires a dataset."""
    with pytest.raises(
        ValueError,
        match=re.escape("The least-squares regression requires data."),
    ):
        PCERegressor(None, probability_space)


def test_discipline_with_least_square(probability_space, dataset, discipline):
    """Check that least square does not require a discipline."""
    with pytest.raises(
        ValueError,
        match=re.escape("The least-squares regression does not require a discipline."),
    ):
        PCERegressor(dataset, probability_space, discipline=discipline)


def test_input_names_with_least_square(dataset, probability_space):
    """Check the input names with least square."""
    assert PCERegressor(dataset, probability_space).input_names == ["x1", "x2"]


def test_input_names_with_quadrature(discipline, probability_space):
    """Check the input names with quadrature."""
    pce = PCERegressor(
        None, probability_space, discipline=discipline, use_quadrature=True
    )
    assert pce.input_names == ["x1", "x2"]


def test_missing_random_variables(dataset):
    """Check that a ValueError is raised when a random variable has no distribution."""
    probability_space = ParameterSpace()
    probability_space.add_random_variable("x1", "SPNormalDistribution")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The probability space does not contain the probability distributions "
            "of the random input variables: x2."
        ),
    ):
        PCERegressor(dataset, probability_space)


@pytest.mark.parametrize("key", ["inputs", "x1", "x2"])
def test_transformer(dataset, probability_space, key):
    """Check that transforming the input data raises an error."""
    with pytest.raises(
        ValueError, match="PCERegressor does not support input transformers."
    ):
        PCERegressor(dataset, probability_space, transformer={key: "MinMaxScaler"})


def test_ot_distribution(dataset):
    """Check that PCERegressor handles only the OTDistribution instances."""
    probability_space = ParameterSpace()
    probability_space.add_random_variable("x1", "SPUniformDistribution")
    probability_space.add_random_variable("x2", "SPUniformDistribution")
    with pytest.raises(
        ValueError,
        match=(
            "The probability distributions of the random variables x1, x2 "
            "are not instances of OTComposedDistribution."
        ),
    ):
        PCERegressor(dataset, probability_space)


def test_initialized_attributes(dataset, probability_space):
    """Check the value of some attributes after instantiation."""
    pce = PCERegressor(dataset, probability_space)
    assert pce._PCERegressor__input_dimension == 2
    assert pce._PCERegressor__cleaning == CleaningOptions()


def test_set_cleaning_options(dataset, probability_space):
    """Check the setting of cleaning options."""
    cleaning_options = CleaningOptions(
        max_considered_terms=128, most_significant=24, significance_factor=1e-3
    )
    pce = PCERegressor(dataset, probability_space, cleaning_options=cleaning_options)
    assert pce._PCERegressor__cleaning == cleaning_options


@pytest.mark.parametrize("use_lars", [False, True])
@pytest.mark.parametrize("use_cleaning", [False, True])
@pytest.mark.parametrize("hyperbolic_parameter", [0.4, 1.0])
def test_learn_linear_model_with_least_square(
    dataset,
    probability_space,
    use_lars,
    use_cleaning,
    hyperbolic_parameter,
):
    """Check the learning stage with least square regression.

    A PCE with a degree equal to 1 shall be able to the linear very precisely whatever
    the algorithms.
    """
    pce = PCERegressor(
        dataset,
        probability_space,
        degree=1,
        use_lars=use_lars,
        use_cleaning=use_cleaning,
        hyperbolic_parameter=hyperbolic_parameter,
    )
    pce.learn()
    assert_almost_equal(pce.predict(array([0.25, 0.75])), array([3.75, -3.75]))


@pytest.mark.parametrize("use_cleaning", [False, True])
@pytest.mark.parametrize("hyperbolic_parameter", [0.4, 1.0])
@pytest.mark.parametrize("dataset_is_none", [False, True])
@pytest.mark.parametrize("n_quadrature_points", [0, 4])
def test_learn_linear_model_with_quadrature_and_discipline(
    discipline,
    probability_space,
    use_cleaning,
    hyperbolic_parameter,
    dataset_is_none,
    n_quadrature_points,
):
    """Check the learning stage with quadrature rule and discipline.

    A PCE with a degree equal to 1 shall be able to the linear very precisely whatever
    the algorithms.
    """
    pce = PCERegressor(
        None if dataset_is_none else IODataset(),
        probability_space,
        discipline=discipline,
        degree=1,
        use_quadrature=True,
        use_cleaning=use_cleaning,
        hyperbolic_parameter=hyperbolic_parameter,
        n_quadrature_points=n_quadrature_points,
    )
    pce.learn()
    assert_almost_equal(pce.predict(array([0.25, 0.75])), array([3.75, -3.75]))


@pytest.mark.parametrize("use_cleaning", [False, True])
@pytest.mark.parametrize("hyperbolic_parameter", [0.4, 1.0])
def test_learn_linear_model_with_quadrature_and_quadrature_points(
    quadrature_points, probability_space, use_cleaning, hyperbolic_parameter
):
    """Check the learning stage with quadrature rule and quadrature points.

    A PCE with a degree equal to 1 shall be able to the linear very precisely whatever
    the algorithms.
    """
    pce = PCERegressor(
        quadrature_points,
        probability_space,
        use_quadrature=True,
        use_cleaning=use_cleaning,
        hyperbolic_parameter=hyperbolic_parameter,
    )
    pce.learn()
    assert_almost_equal(pce.predict(array([0.25, 0.75])), array([3.75, -3.75]))


@pytest.mark.parametrize(
    ("hyperbolic_parameter", "n_coefficients"), [(0.4, 10), (1.0, 20)]
)
def test_learning_hyperbolic_parameter(
    ishigami_dataset, ishigami_probability_space, hyperbolic_parameter, n_coefficients
):
    """Check that the larger the hyperbolic parameter, the fewer the coefficients."""
    pce = PCERegressor(
        ishigami_dataset,
        ishigami_probability_space,
        degree=3,
        hyperbolic_parameter=hyperbolic_parameter,
    )
    pce.learn()
    assert len(pce.algo.getCoefficients()) == n_coefficients


@pytest.mark.parametrize(("use_lars", "n_coefficients"), [(False, 20), (True, 5)])
def test_learning_lars(
    ishigami_dataset, ishigami_probability_space, use_lars, n_coefficients
):
    """Check that the LARS algorithm removes terms to the PCE."""
    pce = PCERegressor(
        ishigami_dataset,
        ishigami_probability_space,
        degree=3,
        use_lars=use_lars,
    )
    pce.learn()
    assert len(pce.algo.getCoefficients()) == n_coefficients


@pytest.mark.parametrize(("use_cleaning", "n_coefficients"), [(False, 20), (True, 5)])
def test_learning_cleaning(
    ishigami_dataset, ishigami_probability_space, use_cleaning, n_coefficients
):
    """Check that the cleaning algorithm removes terms to the PCE."""
    pce = PCERegressor(
        ishigami_dataset,
        ishigami_probability_space,
        degree=3,
        use_cleaning=use_cleaning,
    )
    pce.learn()
    assert len(pce.algo.getCoefficients()) == n_coefficients


__MESSAGE_1 = "max_considered_terms is too important; set it to max_terms."
__MESSAGE_2 = "most_significant is too important; set it to max_considered_terms."


@pytest.mark.parametrize(
    ("max_considered_terms", "most_significant", "messages"),
    [
        (40, 15, [__MESSAGE_1]),
        (40, 21, [__MESSAGE_1, __MESSAGE_2]),
        (18, 21, [__MESSAGE_2]),
    ],
)
def test_learning_cleaning_options_logging(
    ishigami_dataset,
    ishigami_probability_space,
    max_considered_terms,
    most_significant,
    messages,
    caplog,
):
    """Check the messages logged when the cleaning options are inconsistent."""
    caplog.set_level("WARNING")
    pce = PCERegressor(
        ishigami_dataset,
        ishigami_probability_space,
        degree=3,
        use_cleaning=True,
        cleaning_options=CleaningOptions(
            max_considered_terms=max_considered_terms,
            most_significant=most_significant,
        ),
    )
    pce.learn()
    logged_warning_messages = [
        x.message for x in caplog.records if x.levelno == logging.WARNING
    ]
    for message in messages:
        assert message in logged_warning_messages


@pytest.mark.parametrize(("max_considered_terms", "n_coefficients"), [(18, 5), (10, 3)])
def test_learning_cleaning_max_considered_terms(
    ishigami_dataset, ishigami_probability_space, max_considered_terms, n_coefficients
):
    """Check the effect of the cleaning option ``max_considered_terms``."""
    pce = PCERegressor(
        ishigami_dataset,
        ishigami_probability_space,
        degree=3,
        use_cleaning=True,
        cleaning_options=CleaningOptions(
            max_considered_terms=max_considered_terms,
        ),
    )
    pce.learn()
    assert len(pce.algo.getCoefficients()) == n_coefficients


@pytest.mark.parametrize(
    ("most_significant", "n_coefficients"), [(10, [8, 9]), (5, [6])]
)
def test_learning_cleaning_most_significant(
    ishigami_dataset, ishigami_probability_space, most_significant, n_coefficients
):
    """Check the effect of the cleaning option ``most_significant``."""
    pce = PCERegressor(
        ishigami_dataset,
        ishigami_probability_space,
        degree=5,
        use_cleaning=True,
        cleaning_options=CleaningOptions(
            max_considered_terms=50,
            most_significant=most_significant,
        ),
    )
    pce.learn()
    assert len(pce.algo.getCoefficients()) in n_coefficients


@pytest.mark.parametrize(
    ("significance_factor", "n_coefficients"), [(1e-4, [8, 9]), (1e-1, [8])]
)
def test_learning_cleaning_significance_factor(
    ishigami_dataset, ishigami_probability_space, significance_factor, n_coefficients
):
    """Check the effect of the cleaning option ``significance_factor``."""
    pce = PCERegressor(
        ishigami_dataset,
        ishigami_probability_space,
        degree=5,
        use_cleaning=True,
        cleaning_options=CleaningOptions(
            max_considered_terms=50, significance_factor=significance_factor
        ),
    )
    pce.learn()
    assert len(pce.algo.getCoefficients()) in n_coefficients


@pytest.mark.parametrize("after", [False, True])
def test_deepcopy(dataset, probability_space, after):
    """Check that a model can be deepcopied before or after learning."""
    model = PCERegressor(dataset, probability_space)
    if after:
        model.learn()
    model_copy = deepcopy(model)
    if not after:
        model.learn()
        model_copy.learn()

    input_data = {"x1": array([1.0]), "x2": array([2.0])}
    assert_equal(model.predict(input_data), model_copy.predict(input_data))


def test_prediction_jacobian(pce):
    """Check the prediction of the Jacobian."""
    assert compare_dict_of_arrays(
        pce.predict_jacobian({"x1": array([1.0]), "x2": array([2.0])}),
        {
            "y1": {"x1": array([[2.0]]), "x2": array([[3.0]])},
            "y2": {"x1": array([[-2.0]]), "x2": array([[-3.0]])},
        },
        tolerance=1e-3,
    )


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("first", [{"x1": 0.31, "x2": 0.69}, {"x1": 0.31, "x2": 0.69}]),
        (
            "second",
            [{"x1": {"x2": 0}, "x2": {"x1": 0}}, {"x1": {"x2": 0}, "x2": {"x1": 0}}],
        ),
        ("total", [{"x1": 0.31, "x2": 0.69}, {"x1": 0.31, "x2": 0.69}]),
    ],
)
def test_sobol(pce, order, expected):
    """Check the computation of Sobol' indices."""
    computed = getattr(pce, f"{order}_sobol_indices")
    assert len(computed) == len(expected)
    for value1, value2 in zip(computed, expected):
        assert compare_dict_of_arrays(value1, value2, 0.01)


def test_mean_cov_var_std(pce):
    """Check the mean, covariance, variance and standard deviation."""
    vector = FunctionalChaosRandomVector(pce.algo)
    mean = pce.mean
    assert mean.shape == (2,)
    assert_equal(mean, array(vector.getMean()))

    covariance = pce.covariance
    assert covariance.shape == (2, 2)
    assert_equal(covariance, array(vector.getCovariance()))

    variance = pce.variance
    assert variance.shape == (2,)
    assert_equal(variance, covariance.diagonal())

    standard_deviation = pce.standard_deviation
    assert standard_deviation.shape == (2,)
    assert_equal(standard_deviation, variance**0.5)


@pytest.mark.parametrize(
    "name",
    [
        "mean",
        "covariance",
        "variance",
        "standard_deviation",
        "first_sobol_indices",
        "second_sobol_indices",
        "total_sobol_indices",
    ],
)
def test_check_is_trained(untrained_pce, name):
    """Check that a RuntimeError is raised when accessing properties before training."""
    with pytest.raises(
        RuntimeError,
        match=re.escape(f"The PCERegressor must be trained to access {name}."),
    ):
        getattr(untrained_pce, name)


def test_save_load_with_pickle(pce, tmp_wd):
    """Check some attributes are correctly with pickle."""
    with open("model.pkl", "wb") as f:
        dump(pce, f)

    with open("model.pkl", "rb") as f:
        model = load(f)

    assert model._prediction_function
    assert model._mean.size
    assert model._covariance.size
    assert model._variance.size
    assert model._standard_deviation.size
    assert model._first_order_sobol_indices
    assert model._second_order_sobol_indices
    assert model._total_order_sobol_indices


def test_save_load(pce, tmp_wd):
    """Check some attributes are correctly loaded."""
    directory_path = pce.to_pickle("my_model")
    model = import_regression_model(directory_path)
    assert model._prediction_function
    assert model._mean.size
    assert model._covariance.size
    assert model._variance.size
    assert model._standard_deviation.size
    assert model._first_order_sobol_indices
    assert model._second_order_sobol_indices
    assert model._total_order_sobol_indices


def test_multidimensional_variables():
    """Check that a PCERegressor can be built from multidimensional variables."""

    # First,
    # build a PCE of the Ishigami function with 3 scalar inputs.
    def f(x1=0, x2=0, x3=0):
        y = sin(x1) + 7 * sin(x2) ** 2 + 0.1 * x3**4 * sin(x1)
        return y  # noqa: RET504

    discipline = AutoPyDiscipline(f)
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable(
        "x1", "OTUniformDistribution", minimum=-pi, maximum=pi
    )
    parameter_space.add_random_variable(
        "x2", "OTUniformDistribution", minimum=-pi, maximum=pi
    )
    parameter_space.add_random_variable(
        "x3", "OTUniformDistribution", minimum=-pi, maximum=pi
    )

    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y", parameter_space)
    scenario.execute({"algo": "OT_OPT_LHS", "n_samples": 100})
    dataset = scenario.to_dataset(opt_naming=False)

    pce = PCERegressor(dataset, parameter_space)
    pce.learn()
    r2 = R2Measure(pce)
    reference_r2 = r2.compute_learning_measure()
    reference_first_sobol_indices = pce.first_sobol_indices[0]
    reference_second_sobol_indices = pce.second_sobol_indices[0]
    reference_total_sobol_indices = pce.total_sobol_indices[0]

    # Then,
    # build a PCE of the Ishigami function
    # with a 2-length input vector and a scalar input.
    def f(a=array([0, 0]), b=array([0])):  # noqa: B008
        y = array([sin(a[0]) + 7 * sin(a[1]) ** 2 + 0.1 * b[0] ** 4 * sin(a[0])])
        return y  # noqa: RET504

    discipline = AutoPyDiscipline(f, use_arrays=True)
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable(
        "a", "OTUniformDistribution", 2, minimum=-pi, maximum=pi
    )
    parameter_space.add_random_variable(
        "b", "OTUniformDistribution", minimum=-pi, maximum=pi
    )

    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y", parameter_space)
    scenario.execute({"algo": "OT_OPT_LHS", "n_samples": 100})
    dataset = scenario.to_dataset(opt_naming=False)

    pce = PCERegressor(dataset, parameter_space)
    pce.learn()
    r2 = R2Measure(pce)
    assert r2.compute_learning_measure() == reference_r2
    first_sobol_indices = pce.first_sobol_indices[0]
    assert first_sobol_indices["a"] == [
        reference_first_sobol_indices["x1"],
        reference_first_sobol_indices["x2"],
    ]
    assert reference_first_sobol_indices["x3"] == first_sobol_indices["b"]
    second_sobol_indices = pce.second_sobol_indices[0]
    assert second_sobol_indices["a"]["b"] == [
        [
            reference_second_sobol_indices["x1"]["x3"],
        ],
        [
            reference_second_sobol_indices["x2"]["x3"],
        ],
    ]
    assert second_sobol_indices["b"]["a"] == [
        [
            reference_second_sobol_indices["x3"]["x1"],
            reference_second_sobol_indices["x3"]["x2"],
        ],
    ]
    total_sobol_indices = pce.total_sobol_indices[0]
    assert total_sobol_indices["a"] == [
        reference_total_sobol_indices["x1"],
        reference_total_sobol_indices["x2"],
    ]
    assert total_sobol_indices["b"] == reference_total_sobol_indices["x3"]
