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

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import nan
from numpy import ndarray
from numpy.testing import assert_allclose
from scipy.sparse import block_diag

from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.function.evaluation_function import EvaluationFunction
from gemseo.core.function.transformed_input_function import TransformedInputFunction
from gemseo.core.problem.database import Database
from gemseo.core.problem.termination_criterion import DesvarIsNan
from gemseo.core.problem.termination_criterion import FunctionIsNan
from gemseo.space.design import DesignSpace
from gemseo.space.transformation.composition import SpaceComposition
from gemseo.space.transformation.normalization import SpaceNormalization

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


@pytest.fixture
def design_space() -> DesignSpace:
    """A design space with a single variable of two components."""
    space = DesignSpace()
    space.add_real_variable("x", size=2, lower_bound=0.0, upper_bound=10.0)
    return space


@pytest.fixture
def transformation(design_space) -> SpaceComposition:
    """The normalization of the design space."""
    return SpaceComposition(design_space, SpaceNormalization)


@pytest.fixture
def function() -> ArrayFunction:
    """A function whose Jacobian is known."""
    return ArrayFunction(
        lambda x: array([x @ x]), name="f", jac=lambda x: array([2.0 * x]), dim=1
    )


def test_the_argument_is_transformed_not_the_function(function, transformation) -> None:
    """Check that the value is the same in either coordinates."""
    adapted = TransformedInputFunction(EvaluationFunction(function), transformation)

    # [0.1, 0.2] of the working space is [1.0, 2.0] of the original one.
    assert_allclose(adapted.func(array([0.1, 0.2])), function.func(array([1.0, 2.0])))


def test_jacobian_is_mapped_by_the_chain_rule(function, transformation) -> None:
    """Check that the Jacobian is expressed in the working coordinates."""
    adapted = TransformedInputFunction(EvaluationFunction(function), transformation)

    # df/dx = [2, 4] at [1, 2],
    # and the range is 10,
    # so df/dx_working = [20, 40].
    # The shape is the one the evaluation half returns,
    # which flattens the Jacobian of a scalar function,
    # see the test below.
    assert_allclose(adapted.jac(array([0.1, 0.2])), array([20.0, 40.0]))


def test_nan_input_stops_the_evaluation(function, transformation) -> None:
    """Check that a NaN input raises."""
    adapted = TransformedInputFunction(EvaluationFunction(function), transformation)

    with pytest.raises(DesvarIsNan):
        adapted.func(array([nan, 0.2]))


def test_nan_output_stops_the_evaluation(transformation) -> None:
    """Check that a NaN output raises."""
    adapted = TransformedInputFunction(
        EvaluationFunction(ArrayFunction(lambda x: array([nan]), name="f", dim=1)),
        transformation,
    )

    with pytest.raises(FunctionIsNan):
        adapted.func(array([0.1, 0.2]))


def test_nan_output_can_be_tolerated(transformation) -> None:
    """Check that a DOE can turn the stop off."""
    adapted = TransformedInputFunction(
        EvaluationFunction(ArrayFunction(lambda x: array([nan]), name="f", dim=1)),
        transformation,
        stop_if_nan=False,
    )

    assert not adapted.stop_if_nan
    assert isnan_scalar(adapted.func(array([0.1, 0.2])))


def isnan_scalar(value) -> bool:
    """Whether a one-element array holds a NaN.

    Args:
        value: The array.

    Returns:
        Whether it holds a NaN.
    """
    return bool(value[0] != value[0])


def test_the_two_halves_stack(function, transformation, design_space) -> None:
    """Check the whole chain: adaptation on top of recording.

    The algorithm hands a working point,
    the discipline is evaluated at the original one,
    and the database is keyed on the original one.
    """
    database = Database(name="db", input_space=design_space)
    recorded = EvaluationFunction(function, database)
    adapted = TransformedInputFunction(recorded, transformation)

    assert_allclose(adapted.func(array([0.1, 0.2])), array([5.0]))

    # The database is keyed on the original point, not on the working one.
    assert len(database) == 1
    assert_allclose(database.get_function_value("f", array([1.0, 2.0])), array([5.0]))


def test_the_two_halves_flatten_a_scalar_jacobian(
    function, transformation, design_space
) -> None:
    """Check that the flattening for a scalar function survives the split.

    It lives in the evaluation half,
    so the algorithm sees the same shape as before.
    """
    database = Database(name="db", input_space=design_space)
    adapted = TransformedInputFunction(
        EvaluationFunction(function, database), transformation
    )

    assert adapted.jac(array([0.1, 0.2])).shape == (2,)


def test_the_adaptation_half_records_nothing(function, transformation) -> None:
    """Check that the adaptation half knows nothing about a database."""
    adapted = TransformedInputFunction(EvaluationFunction(function), transformation)

    assert not hasattr(adapted, "_database")


def test_an_empty_chain_is_a_pass_through(function, design_space) -> None:
    """Check that an empty composition leaves the coordinates alone.

    This is the shape a problem over a random space takes,
    whose coordinate composition is empty.
    """
    adapted = TransformedInputFunction(
        EvaluationFunction(function), SpaceComposition(design_space)
    )
    input_value = array([1.0, 2.0])

    assert_allclose(adapted.func(input_value), function.func(input_value))
    assert_allclose(adapted.jac(input_value), function.jac(input_value).ravel())


def test_the_count_is_the_one_of_the_half_that_records(
    function, transformation, design_space, enable_function_statistics
) -> None:
    """Check that the count and the counting setting are read from the other half.

    Counting is a recording concern,
    so a driver reading the number of calls of a function
    of the problem it iterates on,
    and a result writing that number back,
    both reach through this half to the one that records.
    """
    recorded = EvaluationFunction(
        function, Database(name="db", input_space=design_space)
    )
    adapted = TransformedInputFunction(recorded, transformation)

    assert adapted.enable_statistics

    adapted.evaluate(array([0.1, 0.2]))
    adapted.evaluate(array([0.3, 0.4]))

    assert adapted.n_calls == 2
    assert recorded.n_calls == 2

    adapted.n_calls = 0

    assert recorded.n_calls == 0


def f_vectorized(x):
    """Compute the sum of the squares of each sample of a matrix.

    Args:
        x: The samples, of shape `(n_samples, 2)`.

    Returns:
        One output value per sample.
    """
    samples = x.reshape((-1, 2))
    return (samples**2).sum(axis=1)


def dfdx_vectorized(x):
    """Differentiate `f_vectorized`, one diagonal block per sample.

    Args:
        x: The samples, of shape `(n_samples, 2)`.

    Returns:
        The block diagonal Jacobian of the samples.
    """
    samples = x.reshape((-1, 2))
    return block_diag([array([2.0 * sample]) for sample in samples], format="csr")


@pytest.fixture
def samples() -> RealArray:
    """Three points of the working space, in the normalized coordinates."""
    return array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])


def test_the_jacobian_of_samples_is_mapped_block_by_block(
    transformation, samples
) -> None:
    """Check that every diagonal block is mapped, not only the first one."""
    function = ArrayFunction(f_vectorized, name="f", jac=dfdx_vectorized, dim=1)
    adapted = TransformedInputFunction(
        EvaluationFunction(function, vectorize=True), transformation
    )
    one_at_a_time = TransformedInputFunction(
        EvaluationFunction(function), transformation
    )

    jacobian = adapted.jac(samples).toarray()
    for index, sample in enumerate(samples):
        assert_allclose(
            jacobian[index, index * 2 : (index + 1) * 2],
            one_at_a_time.jac(sample).toarray()[0],
        )


def test_the_jacobian_of_samples_keeps_its_carrier(transformation, samples) -> None:
    """Check that the mapped Jacobian is carried as the one the function returned."""
    function = ArrayFunction(
        lambda x: f_vectorized(x),
        name="f",
        jac=lambda x: dfdx_vectorized(x).toarray(),
        dim=1,
    )
    adapted = TransformedInputFunction(
        EvaluationFunction(function, vectorize=True), transformation
    )
    assert isinstance(adapted.jac(samples), ndarray)


def test_the_working_database_records_one_entry_per_sample(
    transformation, samples, design_space
) -> None:
    """Check that a vectorized evaluation is recorded sample by sample."""
    working_database = Database(input_space=design_space)
    function = ArrayFunction(f_vectorized, name="f", jac=dfdx_vectorized, dim=1)
    adapted = TransformedInputFunction(
        EvaluationFunction(function, vectorize=True),
        transformation,
        working_database=working_database,
    )
    output_values = adapted.evaluate(samples)
    jacobian = adapted.jac(samples).toarray()

    assert len(working_database) == len(samples)
    for index, sample in enumerate(samples):
        # The keys are the points the algorithm asked for,
        # not the matrix of them,
        # and the values are the ones of the coordinates it works in.
        recorded = working_database[sample]
        assert_allclose(recorded["f"], output_values[index])
        assert_allclose(
            recorded["@f"].toarray()[0],
            jacobian[index, index * 2 : (index + 1) * 2],
        )


def test_a_nan_block_names_the_sample_it_belongs_to(transformation, samples) -> None:
    """Check that a NaN in a block stops the run naming the sample of that block.

    The check belongs to the evaluation half,
    which names the sample in the coordinates the user declared.
    """

    def jac_with_a_nan(x):
        jacobian = dfdx_vectorized(x).toarray()
        jacobian[1, 2:4] = nan
        return jacobian

    function = ArrayFunction(f_vectorized, name="f", jac=jac_with_a_nan, dim=1)
    adapted = TransformedInputFunction(
        EvaluationFunction(function, vectorize=True), transformation
    )
    with pytest.raises(FunctionIsNan, match=r"\[3\. 4\.\]"):
        adapted.jac(samples)
