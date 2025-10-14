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
"""Tests the DOE features related to vectorized computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import column_stack
from numpy import expand_dims
from numpy import hstack
from numpy import vstack
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.sparse import block_diag

from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.scipy.scipy_doe import SciPyDOE
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.chains.chain import MDOChain
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.scenarios.mdo_scenario import MDOScenario

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.evaluation_problem import EvaluationType
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping


def f_vectorized(x: RealArray) -> RealArray:
    """The function computing x_0^2 + x_1^3 in a vectorized way."""
    x = x.reshape((-1, 2))
    y = x[..., [0]] ** 2 + x[..., [1]] ** 3
    return y.ravel()


def dfdx_vectorized(x: RealArray) -> RealArray:
    """The function differentiating x_0^2 + x_1^3 in a vectorized way."""
    reshaped_x = x.reshape((-1, 2))
    n_samples = len(reshaped_x)
    jac = column_stack((2 * reshaped_x[..., 0], 3 * reshaped_x[..., 1] ** 2))
    if n_samples == 1:
        return jac

    return block_diag([array([jac_i]) for jac_i in jac], format="csr")


@pytest.fixture(scope="module", params=(False, True))
def eval_jac(request) -> bool:
    """Whether to evaluate the Jacobian."""
    return request.param


@pytest.fixture(scope="module")
def design_space() -> DesignSpace:
    """The design space."""
    ds = DesignSpace()
    ds.add_variable("in", size=2, lower_bound=0.0, upper_bound=2.0)
    return ds


N_SAMPLES = 3
"""The number of samples used by the DOE algorithms in the test functions."""


@pytest.fixture(scope="module")
def database(design_space, eval_jac) -> Database:
    """The reference database."""
    db = Database(input_space=design_space)
    lib = SciPyDOE("MC")
    samples = lib.compute_doe(design_space, n_samples=N_SAMPLES)
    for sample in samples:
        db.store(sample, {"out": f_vectorized(sample)})
        if eval_jac:
            db.store(sample, {"@out": dfdx_vectorized(sample)})

    return db


class VectorizedDiscipline(Discipline):
    """A differentiated discipline implementing x_0^2 + x_1^3 in a vectorized way."""

    default_grammar_type = Discipline.GrammarType.SIMPLE

    def __init__(self):
        super().__init__()
        self.io.input_grammar.update_from_names(["in"])
        self.io.output_grammar.update_from_names(["out"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {"out": f_vectorized(input_data["in"])}

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        self.jac = {"out": {"in": dfdx_vectorized(self.io.data["in"])}}


class Callback:
    """A callback functor to be evaluated after each evaluation.

    It is used to define the ``callbacks`` field of the ``BaseDOESettings``.
    """

    x: list[tuple[int, EvaluationType]]
    """The sequence of evaluations of the form (index, (output_value, jac_value))."""

    def __init__(self):
        self.x = []

    def __call__(self, index, result):
        self.x.append((index, result))


@pytest.mark.parametrize("preprocess_design_vector", [False, True])
@pytest.mark.parametrize(
    ("design_vector_is_normalized", "design_vectors"),
    [(False, array([[0.0, 1.0], [1.0, 2.0]])), (True, array([[0.0, 0.5], [0.5, 1.0]]))],
)
@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("use_database", [False, True])
def test_preprocess_functions_vectorize(
    design_space,
    preprocess_design_vector,
    design_vector_is_normalized,
    design_vectors,
    vectorize,
    use_database,
):
    """Check the EvaluationProblem.preprocess_functions's argument ``vectorize``.

    Whatever the value of ``vectorize``,
    ``EvaluationProblem.evaluate_functions`` can evaluate a vectorized function.
    """
    if design_vector_is_normalized and not preprocess_design_vector:
        expected = array([0.125, 1.25])
    else:
        expected = array([1.0, 9.0])

    problem = EvaluationProblem(design_space)
    problem.add_observable(MDOFunction(f_vectorized, "out", jac=dfdx_vectorized))
    problem.preprocess_functions(
        use_database=use_database,
        vectorize=vectorize,
        is_function_input_normalized=preprocess_design_vector,
    )
    outputs = problem.evaluate_functions(
        design_vectors,
        output_functions=problem.get_functions(observable_names=())[0],
        design_vector_is_normalized=design_vector_is_normalized,
        preprocess_design_vector=preprocess_design_vector,
    )
    assert_array_equal(outputs[0]["out"], expected)


@pytest.mark.parametrize("vectorize", [False, True])
def test_doe_vectorize_evaluation_problem(
    database, design_space, eval_jac, vectorize, enable_function_statistics
):
    """Check the DOE option 'vectorize' with an EvaluationProblem."""
    problem = EvaluationProblem(design_space)
    problem.add_observable(MDOFunction(f_vectorized, "out", jac=dfdx_vectorized))
    callback = Callback()

    lib = SciPyDOE("MC")
    lib.execute(
        problem,
        n_samples=N_SAMPLES,
        vectorize=vectorize,
        eval_jac=eval_jac,
        callbacks=(callback,),
    )

    # Check the number of calls to the ProblemFunction.
    assert problem.observables[-1].n_calls == (1 if vectorize else N_SAMPLES)

    # Check the evaluations
    # in terms of input values, output values and gradient values.
    assert_frame_equal(
        problem.database.to_dataset(export_gradients=True),
        database.to_dataset(export_gradients=True),
    )

    # Check that the callback has been used correctly.
    # --- Check the indices of the successive evaluations.
    assert [xi[0] for xi in callback.x] == [0, 1, 2]
    # --- Check the output values of the successive evaluations.
    assert_array_equal(
        hstack([xi[1][0]["out"] for xi in callback.x]),
        database.get_function_history("out"),
    )
    # --- Check the Jacobian values of the successive evaluations.
    if eval_jac:
        assert_array_equal(
            expand_dims(vstack([xi[1][1]["out"] for xi in callback.x]), axis=1),
            database.get_gradient_history("out"),
        )
    else:
        assert all(xi[1][1] == {} for xi in callback.x)


@pytest.mark.parametrize("vectorize", [False, True])
def test_doe_vectorize_optimization_problem(
    database, design_space, eval_jac, vectorize, enable_function_statistics
):
    """Check the DOE option 'vectorize' with an OptimizationProblem."""
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(f_vectorized, "out", jac=dfdx_vectorized)

    lib = SciPyDOE("MC")
    lib.execute(problem, n_samples=N_SAMPLES, vectorize=vectorize, eval_jac=eval_jac)

    # Check the number of calls to the ProblemFunction.
    assert problem.objective.n_calls == 1 if vectorize else N_SAMPLES

    # Check the evaluations
    # in terms of input values, output values and gradient values.
    assert_frame_equal(
        problem.database.to_dataset(export_gradients=True),
        database.to_dataset(export_gradients=True),
    )


@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("use_mdo_chain", [False, True])
def test_doe_vectorize_scenario(
    database,
    design_space,
    eval_jac,
    vectorize,
    use_mdo_chain,
    enable_function_statistics,
):
    """Check the DOE option 'vectorize' with an MDOScenario."""
    discipline = VectorizedDiscipline()
    if use_mdo_chain:
        discipline = MDOChain([discipline])

    scenario = MDOScenario(
        [discipline],
        "out",
        design_space,
        formulation_name="DisciplinaryOpt",
    )
    scenario.execute(
        algo_name="MC", n_samples=N_SAMPLES, vectorize=vectorize, eval_jac=eval_jac
    )
    problem = scenario.formulation.optimization_problem

    # Check the number of calls to the ProblemFunction.
    assert problem.objective.n_calls == 1 if vectorize else N_SAMPLES

    # Check the evaluations
    # in terms of input values, output values and gradient values.
    assert_frame_equal(
        problem.database.to_dataset(export_gradients=True),
        database.to_dataset(export_gradients=True),
    )


@pytest.mark.parametrize("eval_jac", [False, True])
@pytest.mark.parametrize("formulation_name", ["DisciplinaryOpt", "MDF"])
@pytest.mark.parametrize("n", [1, 2])
def test_vectorization_sellar(eval_jac, formulation_name, n):
    """Sample Sellar problem in a vectorized way."""
    classes = (Sellar1, Sellar2, SellarSystem)

    # Create the reference results without vectorization.
    scenario = MDOScenario(
        [cls(n=n) for cls in classes],
        "obj",
        SellarDesignSpace(n=n),
        formulation_name=formulation_name,
    )
    scenario.add_constraint("c_1")
    scenario.add_constraint("c_2")
    scenario.execute(
        algo_name="MC", n_samples=N_SAMPLES, vectorize=False, eval_jac=eval_jac
    )
    reference = scenario.formulation.optimization_problem.database.to_dataset(
        export_gradients=True
    )

    # Create the results with vectorization.
    scenario = MDOScenario(
        [cls(n=n) for cls in classes],
        "obj",
        SellarDesignSpace(n=n),
        formulation_name=formulation_name,
    )
    scenario.add_constraint("c_1")
    scenario.add_constraint("c_2")
    scenario.execute(
        algo_name="MC", n_samples=N_SAMPLES, vectorize=True, eval_jac=eval_jac
    )
    result = scenario.formulation.optimization_problem.database.to_dataset(
        export_gradients=True
    )

    # Compare the results
    # in terms of input values, output values and gradient values.
    assert_frame_equal(result, reference)
