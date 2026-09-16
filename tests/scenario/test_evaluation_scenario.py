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

import pytest
from numpy import array
from numpy import sqrt
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.formulation.mdf import MDF
from gemseo.formulation.mdf_settings import MDF_Settings
from gemseo.mda.chain import MDAChain
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.jacobi_settings import MDAJacobi_Settings
from gemseo.scenario.evaluation import EvaluationScenario
from gemseo.scenario.mdo import MDOScenario
from gemseo.space.design import DesignSpace
from gemseo.space.random import RandomSpace
from gemseo.uncertainty.distribution.openturns.triangular_settings import (
    OTTriangularDistribution_Settings,
)
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def discipline_a() -> AnalyticDiscipline:
    """Discipline A."""
    return AnalyticDiscipline({"y": "x+1"}, name="A")


@pytest.fixture
def discipline_b() -> AnalyticDiscipline:
    """Discipline B."""
    return AnalyticDiscipline({"z": "y*2"}, name="B")


@pytest.fixture
def discipline_c() -> AnalyticDiscipline:
    """Discipline C."""
    return AnalyticDiscipline({"w": "x*2"}, name="C")


@pytest.fixture
def design_space() -> DesignSpace:
    """The design space."""
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    return space


@pytest.fixture
def random_space() -> RandomSpace:
    """A random space with an asymmetric triangular variable over [2, 4]."""
    space = RandomSpace()
    space.add_variable(
        "x",
        OTTriangularDistribution_Settings(minimum=2.0, mode=3.5, maximum=4.0),
    )
    return space


def test_random_space_is_used_as_is(discipline_a, random_space):
    """Check that a random space is used as is, without any conversion."""
    scenario = EvaluationScenario([discipline_a], random_space)

    assert scenario.input_space is random_space
    assert scenario.formulation.problem.input_space is random_space
    assert scenario.formulation.problem.database.input_space is random_space
    assert scenario.formulation.variable_sizes == {"x": 1}


def test_random_space_and_normalization(discipline_a, random_space, snapshot):
    """Check that a random space cannot be normalized."""
    scenario = EvaluationScenario([discipline_a], random_space)
    scenario.add_observable("y")
    settings = PYDOE_FULLFACT_Settings(levels=[3], normalize_design_space=True)
    with assert_exception(ValueError, snapshot):
        scenario.execute(settings)


def test_mdo_scenario_requires_a_design_space(discipline_a, random_space, snapshot):
    """Check that an MDO scenario rejects a random space at construction."""
    with assert_exception(TypeError, snapshot):
        MDOScenario([discipline_a], random_space)


def test_random_space_sampling(discipline_a, random_space):
    """Check that a random space is sampled iso-probabilistically."""
    scenario = EvaluationScenario([discipline_a], random_space)
    scenario.add_observable("y")
    scenario.execute(PYDOE_FULLFACT_Settings(levels=[3]))

    # The unit levels 0, 0.5 and 1 are mapped
    # by the inverse CDF of the triangular distribution,
    # which sends 0.5 to the median 2 + sqrt((4 - 2) x (3.5 - 2) / 2),
    # where a geometric mapping would send it to the midpoint 3.
    median = 2.0 + sqrt(1.5)
    dataset = scenario.to_dataset()
    assert_allclose(
        dataset.get_view(variable_names="x"), array([[2.0], [median], [4.0]])
    )
    assert_allclose(
        dataset.get_view(variable_names="y"), array([[3.0], [median + 1.0], [5.0]])
    )


@pytest.mark.parametrize(
    ("differentiation_method", "atol"),
    [
        ("finite_differences", 1e-5),
        ("centered_differences", 1e-8),
        ("complex_step", 1e-12),
    ],
)
def test_approximated_derivatives_over_random_space(
    random_space, differentiation_method, atol
):
    """Check that the derivatives are approximated over a random space.

    A [RandomSpace][gemseo.space.random.RandomSpace] has no current value,
    so
    [set_differentiation_method][gemseo.scenario.evaluation.EvaluationScenario.set_differentiation_method]
    has nothing to cast to `complex128`;
    this cast makes the dtype of the normalization complex,
    which a random space does not need as it cannot be normalized.
    A random space has no bounds to clip the perturbations either,
    as the bounds of a random variable are the limits of the support
    of its probability distribution.
    """
    discipline = AnalyticDiscipline({"y": "x**2"}, name="A")
    scenario = EvaluationScenario([discipline], random_space)
    scenario.add_observable("y")
    scenario.set_differentiation_method(differentiation_method)

    scenario.execute(PYDOE_FULLFACT_Settings(levels=[3], eval_jac=True))

    database = scenario.formulation.problem.database
    assert len(database) == 3
    input_values = array([2.0, 2.0 + sqrt(1.5), 4.0])
    assert_allclose(
        array([value["@y"] for value in database.values()]).ravel(),
        2 * input_values,
        atol=atol,
    )


def test_random_space_seeds_default_input_values(random_space):
    """Check that a random space seeds the default input values.

    The top-level disciplines are seeded with the reference value of the space,
    which is the mean of the probability distributions, so that a default input
    value need not be set discipline by discipline.
    """
    discipline = AnalyticDiscipline({"y": "x+1"}, name="A")

    scenario = EvaluationScenario([discipline], random_space)

    top_level_discipline = scenario.formulation.get_top_level_disciplines()[0]
    mean = random_space.variables["x"].distribution.mean
    assert_allclose(top_level_discipline.io.input_grammar.defaults["x"], mean)


def test_random_space_default_input_values_follow_the_distributions(random_space):
    """Check that the default input values follow the probability distributions.

    The reference value of a random space is read from its distributions, so
    changing them changes the default input values without touching any
    discipline.
    """
    discipline = AnalyticDiscipline({"y": "x+1"}, name="A")
    random_space.remove_variable("x")
    random_space.add_variable(
        "x", OTTriangularDistribution_Settings(minimum=0.0, mode=1.0, maximum=2.0)
    )

    scenario = EvaluationScenario([discipline], random_space)

    top_level_discipline = scenario.formulation.get_top_level_disciplines()[0]
    assert_allclose(top_level_discipline.io.input_grammar.defaults["x"], array([1.0]))


def test_design_space_seeds_default_input_values(design_space):
    """Check that a design space seeds the default input values of the disciplines.

    The reference value of a design space is its current value.
    """
    discipline = AnalyticDiscipline({"y": "x+1"}, name="A")
    design_space.set_current_value({"x": array([0.25])})

    scenario = EvaluationScenario([discipline], design_space)

    top_level_discipline = scenario.formulation.get_top_level_disciplines()[0]
    assert_equal(top_level_discipline.io.input_grammar.defaults["x"], array([0.25]))


def test_two_coupled_disciplines_over_random_space(random_space):
    """Check that strongly coupled disciplines are sampled over a random space.

    A random space seeds the uncertain variables only, so the disciplines
    must still supply the initial couplings the MDA starts from, here the
    default value of `"y2"`.
    """
    disciplines = [
        AnalyticDiscipline({"y1": "x + 0.5 * y2"}, name="A"),
        AnalyticDiscipline({"y2": "0.5 * y1"}, name="B"),
    ]
    scenario = EvaluationScenario(disciplines, random_space)
    assert isinstance(scenario.formulation, MDF)
    assert isinstance(scenario.formulation.mda, MDAChain)

    scenario.add_observable("y1")
    samples = array([[2.0], [4.0]])
    scenario.execute(CustomDOE_Settings(samples=samples))

    # The fixed point of y1 = x + 0.5 * y2 and y2 = 0.5 * y1 is y1 = x / 0.75.
    assert_allclose(scenario.to_dataset().get_view(variable_names="y1"), samples / 0.75)


def test_two_coupled_disciplines_default(discipline_a, discipline_b, design_space):
    """Check the evaluation scenario with two disciplines coupled using default MDA."""
    scenario = EvaluationScenario([discipline_b, discipline_a], design_space)
    assert isinstance(scenario.formulation, MDF)
    assert isinstance(scenario.formulation.mda, MDAChain)
    scenario.add_observable("y")
    scenario.add_observable("z")
    result = scenario.execute(CustomDOE_Settings(samples=array([[2.0], [3.0]])))
    assert result is None
    dataset = scenario.to_dataset()
    assert_equal(dataset.get_view(variable_names="y"), array([[3.0], [4.0]]))
    assert_equal(dataset.get_view(variable_names="z"), array([[6.0], [8.0]]))
    problem = scenario.formulation.problem
    assert isinstance(problem, EvaluationProblem)
    assert problem.function_names == ["y", "z"]
    assert len(problem.database) == 2


def test_two_coupled_disciplines(discipline_a, discipline_b, design_space):
    """Check the evaluation scenario with two disciplines coupled using Jacobi."""
    scenario = EvaluationScenario(
        [discipline_b, discipline_a],
        design_space,
        formulation_settings=MDF_Settings(main_mda_settings=MDAJacobi_Settings()),
    )
    assert isinstance(scenario.formulation, MDF)
    assert isinstance(scenario.formulation.mda, MDAJacobi)
    scenario.add_observable("y")
    scenario.add_observable("z")
    scenario.execute(CustomDOE_Settings(samples=array([[2.0], [3.0]])))
    dataset = scenario.to_dataset()
    assert_equal(dataset.get_view(variable_names="y"), array([[3.0], [4.0]]))
    assert_equal(dataset.get_view(variable_names="z"), array([[6.0], [8.0]]))


def test_observable_name(discipline_a, design_space):
    """Check the use of a custom observable name."""
    scenario = EvaluationScenario([discipline_a], design_space)
    scenario.add_observable("y", observable_name="foo")
    scenario.execute(CustomDOE_Settings(samples=array([[2.0], [3.0]])))
    dataset = scenario.to_dataset()
    assert_equal(dataset.get_view(variable_names="foo"), array([[3.0], [4.0]]))


@pytest.mark.parametrize("add_y", [False, True])
def test_observe_all_outputs(
    discipline_a, discipline_b, discipline_c, design_space, add_y
):
    """Test the observe_all_outputs method."""
    scenario = EvaluationScenario(
        [discipline_b, discipline_a, discipline_c],
        design_space,
    )
    if add_y:
        scenario.add_observable("y")
        assert scenario.formulation.problem.function_names == ["y"]

    scenario.observe_all_outputs()
    assert scenario.formulation.problem.function_names == (
        ["y", "w", "z"]
        if add_y
        else [
            "w",
            "y",
            "z",
        ]
    )


def test_raise_exception_when_missing_algo_settings(discipline_a, design_space):
    """Check that a ValueError is raised when the algo_settings are missing."""
    scenario = EvaluationScenario([discipline_a], design_space)
    msg = (
        "Algorithm settings are necessary for executing a scenario. "
        "Pass the settings in the execute method "
        "or use the set_algorithm method."
    )
    with pytest.raises(ValueError, match=msg):
        scenario.execute()


@pytest.mark.parametrize(
    ("cls", "method"),
    [(EvaluationScenario, "add_observable"), (MDOScenario, "add_objective")],
)
def test_default_input_data(cls, method):
    """The default_input_data argument of EvaluationScenario and MDOScenario
    change the default input of disciplines"""
    disciplines = [
        AnalyticDiscipline({"y": "x+1"}, name="A"),
        AnalyticDiscipline({"z": "y+offset"}, name="B"),
    ]
    assert disciplines[1].io.input_grammar.defaults["offset"] == 0.0

    design_space = DesignSpace()
    design_space.add_variable("x")

    scenario = cls(
        disciplines,
        design_space,
        default_input_data={"offset": array([1.5])},
    )
    getattr(scenario, method)("z")
    scenario.execute(CustomDOE_Settings(samples=array([[2.0], [3.0]])))
    assert disciplines[1].io.input_grammar.defaults["offset"] == 1.5
    assert_equal(
        scenario.formulation.problem.database.get_function_history("z"),
        array([4.5, 5.5]),
    )
