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

import pickle

import pytest
from numpy import array
from numpy import isnan
from numpy import matrix
from numpy import nan
from numpy import ndarray
from numpy.testing import assert_allclose
from scipy.linalg import block_diag
from scipy.sparse import coo_array
from scipy.sparse import coo_matrix
from scipy.sparse import csr_array
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.function.evaluation_function import EvaluationFunction
from gemseo.core.problem.database import Database
from gemseo.core.problem.termination_criterion import DesvarIsNan
from gemseo.core.problem.termination_criterion import FunctionIsNan
from gemseo.space.design import DesignSpace
from gemseo.space.random import RandomSpace
from gemseo.space.transformation._working import create_working_transformation
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.util.derivative.approximator.factory import GradientApproximatorFactory
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def design_space() -> DesignSpace:
    """A design space with a single variable."""
    space = DesignSpace()
    space.add_real_variable("x", size=2, lower_bound=0.0, upper_bound=10.0)
    return space


def compute_squared_norm(input_value):
    """Compute the squared norm of an input value.

    Defined at the module level
    so that a function built on it can be pickled,
    which a process pool requires.

    Args:
        input_value: The input value.

    Returns:
        The squared norm.
    """
    return array([input_value @ input_value])


@pytest.fixture
def function() -> ArrayFunction:
    """A function whose Jacobian is known."""
    return ArrayFunction(
        lambda x: array([x @ x]),
        name="f",
        jac=lambda x: array([2.0 * x]),
        dim=1,
    )


@pytest.fixture
def database(design_space) -> Database:
    """An empty database."""
    return Database(name="db", input_space=design_space)


def test_output_is_recorded(function, database) -> None:
    """Check that an output value is stored under the point it was computed at."""
    recorded = EvaluationFunction(function, database)
    input_value = array([1.0, 2.0])

    assert_allclose(recorded.func(input_value), array([5.0]))
    assert len(database) == 1
    assert_allclose(database.get_function_value("f", input_value), array([5.0]))


def test_output_is_read_back(function, database) -> None:
    """Check that a recorded value is read back instead of recomputed."""
    calls = []
    counting_function = ArrayFunction(
        lambda x: (calls.append(x), array([x @ x]))[1], name="f", dim=1
    )
    recorded = EvaluationFunction(counting_function, database)
    input_value = array([1.0, 2.0])

    recorded.func(input_value)
    recorded.func(input_value)

    assert len(calls) == 1


def test_jacobian_is_recorded(function, database) -> None:
    """Check that a Jacobian is stored under the gradient name."""
    recorded = EvaluationFunction(function, database)
    input_value = array([1.0, 2.0])

    assert_allclose(recorded.jac(input_value), array([2.0, 4.0]))
    assert_allclose(
        database.get_function_value(Database.get_gradient_name("f"), input_value),
        array([[2.0, 4.0]]),
    )


def test_jacobian_is_not_recorded_when_asked(function, database) -> None:
    """Check that the Jacobian storage can be turned off."""
    recorded = EvaluationFunction(function, database, store_jacobian=False)
    input_value = array([1.0, 2.0])

    recorded.jac(input_value)

    assert (
        database.get_function_value(Database.get_gradient_name("f"), input_value)
        is None
    )


def test_approximated_jacobian_does_not_record_its_perturbations(
    function, database, design_space
) -> None:
    """Check the invariant that matters most for the history.

    An approximated Jacobian must add one gradient entry to the database,
    and its finite-difference perturbations must add none.
    """
    recorded = EvaluationFunction(
        function,
        database,
        design_space=design_space,
        differentiation_method="finite_differences",
        differentiation_method_options={"step": 1e-7},
    )
    input_value = array([1.0, 2.0])

    assert_allclose(recorded.jac(input_value), array([2.0, 4.0]), rtol=1e-5)

    # One entry, at the point asked for, not one per perturbation.
    assert len(database) == 1
    assert_allclose(
        database.get_function_value(Database.get_gradient_name("f"), input_value),
        array([[2.0, 4.0]]),
        rtol=1e-5,
    )


def test_jacobian_approximator_is_built_lazily(
    function, design_space, monkeypatch
) -> None:
    """Check that the Jacobian approximator is built once, on first use.

    A driver installs a transformation and releases it again
    before ever asking for a Jacobian,
    at every sub-scenario execution of a bi-level formulation,
    so building the approximator eagerly there,
    and again on release,
    would build two that are never used.
    """
    calls = []
    original_create = GradientApproximatorFactory.create

    def counting_create(self, *args, **kwargs):
        calls.append(1)
        return original_create(self, *args, **kwargs)

    monkeypatch.setattr(GradientApproximatorFactory, "create", counting_create)

    # No database, so a second `jac` call cannot be answered from a database
    # read instead of the approximator, keeping this test about the
    # approximator's own cache.
    recorded = EvaluationFunction(
        function,
        None,
        design_space=design_space,
        differentiation_method="finite_differences",
        differentiation_method_options={"step": 1e-7},
    )
    transformation = create_working_transformation(design_space, normalize=True)

    recorded.set_perturbation_transformation(transformation)
    recorded.set_perturbation_transformation(None)

    assert not calls

    input_value = array([1.0, 2.0])
    recorded.jac(input_value)

    assert len(calls) == 1

    recorded.jac(input_value)

    assert len(calls) == 1


def test_no_database_still_evaluates(function) -> None:
    """Check that a `None` database disables the store and nothing else."""
    recorded = EvaluationFunction(function, None)
    input_value = array([1.0, 2.0])

    assert_allclose(recorded.func(input_value), array([5.0]))
    assert_allclose(recorded.jac(input_value), array([2.0, 4.0]))


def test_no_database_still_approximates(function, design_space) -> None:
    """Check that the approximation does not depend on the database."""
    recorded = EvaluationFunction(
        function,
        None,
        design_space=design_space,
        differentiation_method="finite_differences",
        differentiation_method_options={"step": 1e-7},
    )

    assert_allclose(recorded.jac(array([1.0, 2.0])), array([2.0, 4.0]), rtol=1e-5)


def test_approximated_jacobian_with_no_design_space_and_a_transformation(
    function,
) -> None:
    """Check that a transformation over a non-design space leaves perturbations unbounded.

    `design_space` is `None` when the input space is not a design space,
    e.g. a `RandomSpace`;
    the approximator then takes the perturbations without bounds,
    whatever the working space of the transformation is.
    """  # noqa: E501
    space = RandomSpace()
    space.add_variable("x", OTUniformDistribution_Settings())
    recorded = EvaluationFunction(
        function,
        None,
        differentiation_method="finite_differences",
        differentiation_method_options={"step": 1e-7},
    )
    recorded.set_perturbation_transformation(create_working_transformation(space))

    assert_allclose(recorded.jac(array([1.0, 2.0])), array([2.0, 4.0]), rtol=1e-5)


def test_nan_output_is_recorded_before_stopping(database, snapshot) -> None:
    """Check that a NaN output reaches the database before the evaluation stops.

    The point the evaluation went wrong at is recorded first,
    so that it is visible in the history,
    and the stop follows.
    This half performs the check as well as the adaptation one,
    since a driver deriving no problem,
    such as a linear one or a meta-algorithm,
    goes through this half only.
    """
    recorded = EvaluationFunction(
        ArrayFunction(lambda x: array([nan]), name="f", dim=1), database
    )
    input_value = array([1.0, 2.0])

    with assert_exception(FunctionIsNan, snapshot):
        recorded.func(input_value)

    assert len(database) == 1


def test_nan_output_can_be_tolerated(database) -> None:
    """Check that a problem tolerating a NaN records it and does not stop."""
    recorded = EvaluationFunction(
        ArrayFunction(lambda x: array([nan]), name="f", dim=1),
        database,
        stop_if_nan=False,
    )

    recorded.func(array([1.0, 2.0]))

    assert len(database) == 1


def test_nan_input_stops_the_output(database, snapshot) -> None:
    """Check that a NaN input stops before the function and the database.

    This half performs the check as well as the adaptation one,
    since a driver deriving no problem,
    such as a linear one or a meta-algorithm,
    goes through this half only,
    and so does a caller attaching a recording without transforming the problem.
    """
    calls = []
    recorded = EvaluationFunction(
        ArrayFunction(lambda x: (calls.append(x), array([0.0]))[1], name="f", dim=1),
        database,
    )

    with assert_exception(DesvarIsNan, snapshot):
        recorded.func(array([1.0, nan]))

    assert not calls
    assert not len(database)


def test_nan_input_stops_the_jacobian(database, snapshot) -> None:
    """Check that a NaN input stops before the derivatives and the database."""
    calls = []
    recorded = EvaluationFunction(
        ArrayFunction(
            lambda x: array([0.0]),
            name="f",
            jac=lambda x: (calls.append(x), array([[0.0, 0.0]]))[1],
            dim=1,
        ),
        database,
    )

    with assert_exception(DesvarIsNan, snapshot):
        recorded.jac(array([1.0, nan]))

    assert not calls
    assert not len(database)


def test_nan_input_stops_even_without_database(snapshot) -> None:
    """Check that the check does not depend on the database."""
    recorded = EvaluationFunction(
        ArrayFunction(lambda x: array([0.0]), name="f", dim=1), None
    )

    with assert_exception(DesvarIsNan, snapshot):
        recorded.func(array([1.0, nan]))


def compute_block_diagonal_jacobian(input_values, nan_index: int | None = 1):
    """Compute the block diagonal Jacobian of a matrix of samples.

    By default the block of the second sample holds a NaN,
    so that the check has one block to single out among several.

    Args:
        input_values: The samples.
        nan_index: The index of the sample whose block holds a NaN.
            If `None`, no block holds one.

    Returns:
        The block diagonal Jacobian.
    """
    return block_diag(*[
        array([
            [
                nan if index == nan_index else 2.0 * input_value[0],
                2.0 * input_value[1],
            ]
        ])
        for index, input_value in enumerate(input_values)
    ])


def build_vectorized_function(jacobian_function) -> ArrayFunction:
    """Build a vectorized function with a given Jacobian.

    Args:
        jacobian_function: The function computing the block diagonal Jacobian.

    Returns:
        The vectorized function.
    """
    return ArrayFunction(
        lambda input_values: (input_values * input_values).sum(1),
        name="f",
        jac=jacobian_function,
        dim=1,
    )


@pytest.fixture
def samples():
    """Three samples of the design space."""
    return array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.fixture
def vectorized_function() -> ArrayFunction:
    """A vectorized function whose Jacobian holds a NaN for the second sample."""
    return build_vectorized_function(compute_block_diagonal_jacobian)


@pytest.mark.parametrize("store_jacobian", [False, True])
def test_vectorized_nan_jacobian_stops_the_run(
    vectorized_function, database, design_space, samples, store_jacobian, snapshot
) -> None:
    """Check that a NaN in a block of a vectorized Jacobian stops the evaluation.

    The message names the sample the block belongs to
    rather than the whole block diagonal matrix.
    A run that records the blocks keeps every one of them before the stop,
    as in the output path:
    the matrix is computed by then,
    so the history keeps it.
    A run that records none stops all the same,
    the check being the business of the evaluation
    rather than of the recording.
    """
    recorded = EvaluationFunction(
        vectorized_function,
        database,
        store_jacobian=store_jacobian,
        vectorize=True,
        design_space=design_space,
    )

    with assert_exception(FunctionIsNan, snapshot):
        recorded.jac(samples)

    if not store_jacobian:
        assert len(database) == 0
        return

    gradient_name = Database.get_gradient_name("f")
    assert len(database) == 3
    assert isnan(database.get_function_value(gradient_name, samples[1])).any()
    assert_allclose(
        database.get_function_value(gradient_name, samples[2]), array([[10.0, 12.0]])
    )


def test_vectorized_nan_jacobian_stops_the_run_without_database(
    vectorized_function, samples, snapshot
) -> None:
    """Check that the block by block check needs no database.

    The check belongs to the evaluation rather than to the recording,
    so a driver that records nothing is stopped by the same message,
    naming the sample the block belongs to.
    """
    recorded = EvaluationFunction(vectorized_function, None, vectorize=True)

    with assert_exception(FunctionIsNan, snapshot):
        recorded.jac(samples)


def test_vectorized_nan_jacobian_stops_the_run_without_input_space(
    vectorized_function, database, samples, snapshot
) -> None:
    """Check that the check of a vectorized Jacobian needs no input space.

    A block is as wide as a sample,
    so the width comes from the samples the call is handed
    rather than from a space declared at build time:
    a function built without one cuts the very same blocks,
    records them and stops on the NaN.
    """
    recorded = EvaluationFunction(vectorized_function, database, vectorize=True)

    with assert_exception(FunctionIsNan, snapshot):
        recorded.jac(samples)

    assert_allclose(
        database.get_function_value(Database.get_gradient_name("f"), samples[2]),
        array([[10.0, 12.0]]),
    )


def test_vectorized_densified_legacy_sparse_jacobian_holds_no_nan(
    database, design_space, samples
) -> None:
    """Check that a densified legacy sparse Jacobian without a NaN passes the check.

    A driver without sparse support asks for the densification,
    and a legacy `scipy.sparse` matrix densifies to a `numpy.matrix`,
    which collapses the blocks the check gathers back to two dimensions.
    The check reads them from a base array view for that reason,
    and what the caller gets keeps its type.
    """
    recorded = EvaluationFunction(
        build_vectorized_function(
            lambda input_values: csr_matrix(
                compute_block_diagonal_jacobian(input_values, None)
            )
        ),
        database,
        support_sparse_jacobian=False,
        vectorize=True,
        design_space=design_space,
    )

    jacobians = recorded.jac(samples)

    assert not isnan(jacobians).any()
    assert isinstance(jacobians, matrix)
    assert len(database) == 3
    stored = database.get_function_value(Database.get_gradient_name("f"), samples[1])
    assert isinstance(stored, matrix)
    assert_allclose(stored, array([[6.0, 8.0]]))


def test_vectorized_densified_legacy_sparse_nan_jacobian_stops_the_run(
    database, design_space, samples, snapshot
) -> None:
    """Check that a densified legacy sparse Jacobian holding a NaN stops the run.

    The message names the sample the block belongs to,
    exactly as it does for a Jacobian that was dense to begin with.
    """
    recorded = EvaluationFunction(
        build_vectorized_function(
            lambda input_values: csr_matrix(
                compute_block_diagonal_jacobian(input_values)
            )
        ),
        database,
        support_sparse_jacobian=False,
        vectorize=True,
        design_space=design_space,
    )

    with assert_exception(FunctionIsNan, snapshot):
        recorded.jac(samples)


def test_vectorized_nan_jacobian_can_be_tolerated(
    vectorized_function, database, design_space, samples
) -> None:
    """Check that a problem tolerating a NaN records the blocks and does not stop."""
    recorded = EvaluationFunction(
        vectorized_function,
        database,
        vectorize=True,
        design_space=design_space,
        stop_if_nan=False,
    )

    jacobians = recorded.jac(samples)

    assert isnan(jacobians).any()
    assert len(database) == 3


def test_vectorized_sparse_jacobian_is_not_checked(
    database, design_space, samples
) -> None:
    """Check that a sparse vectorized Jacobian is left unchecked.

    `check_for_nan` reads no sparse matrix,
    here no more than in the non-vectorized path:
    a driver declaring it supports one reads the matrix itself.
    """
    recorded = EvaluationFunction(
        build_vectorized_function(
            lambda input_values: csr_array(
                compute_block_diagonal_jacobian(input_values)
            )
        ),
        database,
        vectorize=True,
        design_space=design_space,
    )

    jacobians = recorded.jac(samples)

    assert isinstance(jacobians, csr_array)
    assert len(database) == 3


@pytest.mark.parametrize(
    ("sparse_class", "support_sparse_jacobian", "jacobian_class", "block_class"),
    [
        (coo_matrix, True, coo_matrix, csr_matrix),
        (coo_matrix, False, matrix, matrix),
        (coo_array, True, coo_array, csr_array),
        (coo_array, False, ndarray, ndarray),
    ],
)
def test_vectorized_coo_jacobian_is_recorded(
    database,
    design_space,
    samples,
    sparse_class,
    support_sparse_jacobian,
    jacobian_class,
    block_class,
) -> None:
    """Check that a vectorized Jacobian carried by a COO matrix is recorded.

    A COO matrix is meant for assembling a matrix
    rather than for reading one
    and `scipy.sparse` refuses to index it,
    so the blocks recorded for the samples are cut from a CSR copy of it.
    The matrix the caller is handed is the one the function computed,
    in the format it chose,
    and a driver without sparse support is handed a dense one
    as it is for any other format.
    """
    recorded = EvaluationFunction(
        build_vectorized_function(
            lambda input_values: sparse_class(
                compute_block_diagonal_jacobian(input_values, None)
            )
        ),
        database,
        support_sparse_jacobian=support_sparse_jacobian,
        vectorize=True,
        design_space=design_space,
    )

    jacobians = recorded.jac(samples)

    assert isinstance(jacobians, jacobian_class)
    assert len(database) == 3
    stored = database.get_function_value(Database.get_gradient_name("f"), samples[1])
    assert isinstance(stored, block_class)
    assert_allclose(
        stored.todense() if issparse(stored) else stored, array([[6.0, 8.0]])
    )


@pytest.mark.parametrize("use_database", [False, True])
def test_sparse_jacobian_is_not_flattened(database, use_database) -> None:
    """Check that a sparse Jacobian of a function of dimension 1 keeps its type.

    A driver declaring it supports a sparse Jacobian asks for no densification
    and reads the matrix itself,
    so flattening it would hand that driver an array of another type and another shape.
    """
    recorded = EvaluationFunction(
        ArrayFunction(
            lambda x: array([x.sum()]),
            name="f",
            jac=lambda x: csr_array([[1.0, 1.0]]),
            dim=1,
        ),
        database if use_database else None,
    )

    jacobian = recorded.jac(array([1.0, 2.0]))

    assert isinstance(jacobian, csr_array)
    assert jacobian.shape == (1, 2)


def test_pre_compute_at_new_point(function, database) -> None:
    """Check that the hook fires on a point absent from the database."""
    new_points = []
    recorded = EvaluationFunction(function, database)
    recorded.pre_compute_at_new_point = lambda: new_points.append(1)

    recorded.func(array([1.0, 2.0]))
    recorded.func(array([1.0, 2.0]))
    recorded.func(array([3.0, 4.0]))

    assert len(new_points) == 2


def test_original_is_the_wrapped_function(function, database) -> None:
    """Check that the recording wrapper does not hide the user's function."""
    recorded = EvaluationFunction(function, database)
    recorded.original = function.original

    assert recorded.original is function


def test_approximated_jacobian_with_a_transformation_can_be_pickled() -> None:
    """Check that a wrapper told a transformation survives a process pool.

    The approximator holds a method of the wrapper,
    and pickle reduces a bound method through the name it carries,
    so a mangled name would be missing from the class
    when a worker process loads the wrapper back;
    the transformation it was told has to make the round trip too,
    so that a worker perturbs the working point the same way.
    """
    space = DesignSpace()
    space.add_real_variable("x", size=2, lower_bound=0.0, upper_bound=10.0)
    recorded = EvaluationFunction(
        ArrayFunction(compute_squared_norm, name="f", dim=1),
        None,
        design_space=space,
        differentiation_method="finite_differences",
        differentiation_method_options={"step": 1e-7},
    )
    recorded.set_perturbation_transformation(
        create_working_transformation(space, normalize=True)
    )
    input_value = array([1.0, 2.0])

    unpickled = pickle.loads(pickle.dumps(recorded))

    assert_allclose(unpickled.func(input_value), recorded.func(input_value))
    assert_allclose(unpickled.jac(input_value), array([2.0, 4.0]), atol=1e-5)


def test_approximated_jacobian_already_built_can_be_pickled() -> None:
    """Check that a wrapper whose approximator was already used can be pickled.

    The approximator is cached lazily, on the first `jac` call,
    so this exercises the round trip of a wrapper
    that already holds one,
    as opposed to
    [test_approximated_jacobian_with_a_transformation_can_be_pickled][],
    which pickles one before it is ever built.
    """
    space = DesignSpace()
    space.add_real_variable("x", size=2, lower_bound=0.0, upper_bound=10.0)
    recorded = EvaluationFunction(
        ArrayFunction(compute_squared_norm, name="f", dim=1),
        None,
        design_space=space,
        differentiation_method="finite_differences",
        differentiation_method_options={"step": 1e-7},
    )
    input_value = array([1.0, 2.0])

    assert_allclose(recorded.jac(input_value), array([2.0, 4.0]), atol=1e-5)

    unpickled = pickle.loads(pickle.dumps(recorded))

    assert_allclose(unpickled.jac(input_value), array([2.0, 4.0]), atol=1e-5)


def test_no_counter_is_built_when_the_calls_are_not_counted(function) -> None:
    """Check that a function not counting its calls builds no counter.

    A counter is a piece of shared memory with a lock of its own,
    and a driver rebuilds every wrapper of a problem
    each time it wraps the functions,
    so a counter that nothing reads is paid for at every run.
    """
    assert not EvaluationFunction.enable_statistics

    recorded = EvaluationFunction(function)

    assert "_n_calls" not in recorded.__dict__
    assert recorded.n_calls == 0


def test_the_count_survives_a_pickle_round_trip(enable_function_statistics) -> None:
    """Check that a wrapper unpickled in a worker carries on from its count."""
    recorded = EvaluationFunction(ArrayFunction(compute_squared_norm, name="f", dim=1))
    recorded.evaluate(array([1.0, 2.0]))
    recorded.evaluate(array([3.0, 4.0]))

    unpickled = pickle.loads(pickle.dumps(recorded))

    assert unpickled.n_calls == 2

    unpickled.evaluate(array([5.0, 6.0]))

    assert unpickled.n_calls == 3


def test_the_count_survives_a_pickle_round_trip_when_counting_starts_late() -> None:
    """Check the count of a wrapper built before the counters were enabled.

    Whether the calls are counted is a setting a user changes at any time,
    so a wrapper built while the counters were disabled carries no counter
    and has to build one,
    both when it starts counting and when it is unpickled,
    the state it is restored from carrying the count as a plain integer.
    """
    recorded = EvaluationFunction(ArrayFunction(compute_squared_norm, name="f", dim=1))
    assert "_n_calls" not in recorded.__dict__

    recorded.enable_statistics = True
    recorded.evaluate(array([1.0, 2.0]))
    recorded.evaluate(array([3.0, 4.0]))

    assert recorded.n_calls == 2

    unpickled = pickle.loads(pickle.dumps(recorded))

    assert unpickled.n_calls == 2

    unpickled.evaluate(array([5.0, 6.0]))

    assert unpickled.n_calls == 3
