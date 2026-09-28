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

"""Tests of the store of the evaluations in the working coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest
from numpy import array
from numpy.testing import assert_array_equal

from gemseo import configuration
from gemseo import execute_algo
from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.doe.pydoe.settings.pydoe_lhs import PYDOE_LHS_Settings
from gemseo.optimization.augmented_lagrangian.settings.order_1 import (
    Augmented_Lagrangian_Order_1_Settings,
)
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.optimization.power_2 import Power2
from gemseo.space.design import DesignSpace
from gemseo.space.transformation._working import create_working_transformation
from gemseo.util.global_configuration import GlobalConfiguration

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gemseo.core.problem.database import Database


@pytest.fixture
def working_database_is_enabled() -> Iterator[None]:
    """Record the evaluations in the working coordinates."""
    EvaluationProblem.enable_working_database = True
    yield
    EvaluationProblem.enable_working_database = False


SETTINGS = {
    # A gradient-based run, whose Jacobians are recorded too.
    "gradient_based": ("opt", SLSQP_Settings(max_iter=5)),
    # A sampling run, which evaluates without iterating.
    "doe": ("doe", PYDOE_LHS_Settings(n_samples=4, seed=1)),
    # Nested drivers: this one hands sub-problems to a sub-driver,
    # which builds a working problem for each of them in turn.
    "nested": (
        "opt",
        Augmented_Lagrangian_Order_1_Settings(
            max_iter=3, sub_algorithm_settings=SLSQP_Settings(max_iter=3)
        ),
    ),
}


def dump(database: Database) -> list[tuple[tuple[float, ...], list[Any]]]:
    """Return the content of a database, in a comparable form.

    Args:
        database: The database to dump.

    Returns:
        The input value and the output values of each entry.
    """
    return [
        (
            tuple(input_value.unwrap().tolist()),
            sorted((name, str(value)) for name, value in database[input_value].items()),
        )
        for input_value in database
    ]


@pytest.mark.parametrize(("algo_type", "settings"), SETTINGS.values(), ids=SETTINGS)
def test_the_store_changes_nothing(algo_type, settings) -> None:
    """Check that a run gives the same results whether the store is on or off.

    The whole value of this store is that it observes a run without taking part in it,
    so a bug being hunted must not move
    when it is switched on.
    """
    reference_problem = Power2()
    reference_result = execute_algo(
        reference_problem, algo_type=algo_type, settings_model=settings
    )

    EvaluationProblem.enable_working_database = True
    try:
        problem = Power2()
        result = execute_algo(problem, algo_type=algo_type, settings_model=settings)
    finally:
        EvaluationProblem.enable_working_database = False

    assert dump(problem.database) == dump(reference_problem.database)
    assert str(result) == str(reference_result)


@pytest.mark.usefixtures("working_database_is_enabled")
def test_no_store_for_a_meta_algorithm() -> None:
    """Check that an algorithm working in the user's coordinates records nothing.

    A meta-algorithm builds no working problem of its own:
    it builds sub-problems and hands them to a sub-driver,
    which builds a working problem for each of them in turn.
    There is therefore no point of an algorithm to record at this level,
    and a second store in the coordinates the user declared
    would only duplicate the one they read.
    """
    problem = Power2()
    execute_algo(
        problem,
        settings_model=Augmented_Lagrangian_Order_1_Settings(
            max_iter=3, sub_algorithm_settings=SLSQP_Settings(max_iter=3)
        ),
    )

    assert problem.working_database is None


def test_no_store_by_default() -> None:
    """Check that nothing is recorded unless the store is asked for."""
    problem = Power2()
    execute_algo(problem, settings_model=SLSQP_Settings(max_iter=3))

    assert problem.working_database is None


@pytest.mark.usefixtures("working_database_is_enabled")
def test_the_store_holds_the_points_of_the_algorithm() -> None:
    """Check what the store holds, against the one the user reads."""
    problem = Power2()
    execute_algo(problem, settings_model=SLSQP_Settings(max_iter=3))

    working_database = problem.working_database
    assert working_database.name == f"{problem.database.name}_working"
    # The design space of Power2 is [-1, 1]^3
    # and the algorithm normalizes it,
    # so the two stores hold the same points expressed in their own coordinates.
    assert len(working_database) == len(problem.database)
    for working, original in zip(working_database, problem.database, strict=True):
        assert_array_equal(working.unwrap(), (original.unwrap() + 1.0) / 2.0)
        assert sorted(working_database[working]) == sorted(problem.database[original])


@pytest.mark.usefixtures("working_database_is_enabled")
def test_the_store_records_without_the_shared_one() -> None:
    """Check that the store is independent of the database of the problem.

    Disabling the database of the problem is precisely
    when someone hunting a bug wants this store,
    so the two settings do not overlap.
    """
    problem = Power2()
    execute_algo(problem, settings_model=SLSQP_Settings(max_iter=3, use_database=False))

    assert len(problem.database) == 0
    assert len(problem.working_database) > 0


@pytest.mark.usefixtures("working_database_is_enabled")
def test_the_store_holds_the_jacobians_of_the_algorithm() -> None:
    """Check that a Jacobian is recorded in the coordinates it is asked in."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    problem = EvaluationProblem(design_space)
    problem.add_observable(
        ArrayFunction(lambda x: 2.0 * x, name="f", jac=lambda x: array([[2.0]]))
    )
    chain = create_working_transformation(design_space, normalize=True)
    problem.bind_functions()
    working_problem = problem.create_working_problem(chain)

    working_problem.observables[0].jac(array([0.75]))

    (entry,) = problem.working_database
    assert_array_equal(entry.unwrap(), array([0.75]))
    # The chain rule scales the Jacobian by the range of the variable.
    assert_array_equal(problem.working_database[entry]["@f"], array([[4.0]]))
    # The store of the user holds the same evaluation in their own coordinates.
    ((original_entry, _),) = dump(problem.database)
    assert original_entry == (0.5,)


@pytest.mark.usefixtures("restore_configuration_options")
@pytest.mark.parametrize(
    ("configuration_settings", "expected"),
    [
        ({}, False),
        ({"enable_working_database": True}, True),
    ],
)
def test_the_configuration_drives_the_flag(configuration_settings, expected) -> None:
    """Check the setting of the global configuration against the flag it writes."""
    settings = GlobalConfiguration(**configuration_settings)
    assert settings.enable_working_database is expected
    assert EvaluationProblem.enable_working_database is expected


@pytest.mark.usefixtures("restore_configuration_options")
def test_the_fast_mode_drives_the_flag() -> None:
    """Check that the fast mode switches the store off.

    A debugging aid is one of the things the fast mode switches off,
    and leaving the fast mode resets the store to its default,
    which is off.
    """
    configuration.enable_working_database = True

    configuration.enable_fast_mode()
    assert configuration.enable_working_database is False
    assert EvaluationProblem.enable_working_database is False

    configuration.disable_fast_mode()
    assert configuration.enable_working_database is False
    assert EvaluationProblem.enable_working_database is False


def test_the_store_is_dropped_when_the_setting_is_turned_off() -> None:
    """Check the store of a run made after the setting was turned off.

    Assigning the store only when the setting is on
    left the one of an earlier run in place,
    so the later run kept writing into it.
    """
    problem = Power2()
    try:
        configuration.enable_working_database = True
        execute_algo(problem, settings_model=SLSQP_Settings(max_iter=3))
        first_store = problem.working_database
        first_length = len(first_store)

        configuration.enable_working_database = False
        problem.reset()
        # A different starting point,
        # so that the second run visits points the first one did not
        # and a store it kept writing into would grow.
        problem.design_space.set_current_value(array([0.5, 0.5, 0.5]))
        execute_algo(problem, settings_model=SLSQP_Settings(max_iter=3))
    finally:
        configuration.enable_working_database = False

    assert first_length > 0
    assert len(first_store) == first_length
    assert problem.working_database is None
