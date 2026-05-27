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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import concatenate
from numpy import full
from numpy import hstack
from numpy import zeros
from numpy.testing import assert_allclose
from pandas._testing import assert_frame_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.scipy.settings.mc import MC_Settings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.formulations.factory import MDO_FORMULATION_FACTORY
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.problems.mdo.sellar.utils import WITH_2D_ARRAY
from gemseo.problems.mdo.sellar.variables import C_1
from gemseo.problems.mdo.sellar.variables import C_2
from gemseo.problems.mdo.sellar.variables import OBJ
from gemseo.problems.mdo.sellar.variables import X_1
from gemseo.problems.mdo.sellar.variables import X_2
from gemseo.problems.mdo.sellar.variables import X_SHARED
from gemseo.problems.mdo.sellar.variables import Y_1
from gemseo.problems.mdo.sellar.variables import Y_2
from gemseo.scenarios.mdo import MDOScenario
from gemseo.utils.pickle import to_pickle

if TYPE_CHECKING:
    from numpy import ndarray


@pytest.fixture(params=[1, 2])
def n(request) -> int:
    """The dimension of the local design variables and coupling variables."""
    return request.param


@pytest.fixture(params=[SellarSystem, Sellar1, Sellar2])
def discipline(request, n: int) -> SellarSystem | Sellar1 | Sellar2:
    """A Sellar discipline."""
    return request.param(n=n)


@pytest.fixture
def disciplines(n: int) -> tuple[SellarSystem, Sellar1, Sellar2]:
    """The Sellar disciplines."""
    return SellarSystem(n=n), Sellar1(n=n), Sellar2(n=n)


@pytest.fixture
def input_data(n: int) -> dict[str, ndarray]:
    """Generate a point at which the problem is linearized."""
    x_shared = [1.2, 3.4]
    if WITH_2D_ARRAY:  # pragma: no cover
        x_shared = [x_shared]
    return {
        X_1: full(n, 2.1),
        X_2: full(n, 2.1),
        X_SHARED: array(x_shared),
        Y_1: full(n, 2.135),
        Y_2: full(n, 3.584),
    }


@pytest.fixture
def output_data(
    discipline: SellarSystem | Sellar1 | Sellar2, n: int
) -> dict[str, ndarray]:
    """The default output data."""
    if isinstance(discipline, SellarSystem):
        return {OBJ: array([1.36787944]), C_1: full(n, 2.16), C_2: full(n, -23.0)}
    if isinstance(discipline, Sellar1):
        return {Y_1: full(n, 0.89442719)}

    return {Y_2: full(n, 2.0)}


@pytest.fixture
def x_opt(n: int) -> ndarray:
    """The optimal design vector."""
    return hstack((zeros(0), array((1.9776, 0.0))))


def test_execution(sellar_with_2d_array, discipline, output_data, n) -> None:
    """Check the output data of the Sellar disciplines with default input values."""
    discipline.execute()
    for output_name, output_value in output_data.items():
        assert_allclose(discipline.io.output_data[output_name], output_value, rtol=1e-8)


def test_linearization(sellar_with_2d_array, discipline, input_data) -> None:
    """Check the Jacobian value of the Sellar discipline with default input values."""
    assert discipline.check_jacobian(
        input_data, derr_approx=discipline.LinearizationMode.COMPLEX_STEP, step=1e-30
    )


def test_serialize(sellar_with_2d_array, discipline, tmp_wd) -> None:
    """Verify the serialization."""
    file_path = Path("discipline.pkl")
    to_pickle(discipline, file_path)
    assert file_path.exists()


@pytest.mark.parametrize(
    ("cls", "options"),
    [
        (
            MDAGaussSeidel,
            {"max_mda_iter": 100, "tolerance": 1e-14, "over_relaxation_factor": 0.99},
        ),
        (MDAJacobi, {"tolerance": 1e-14}),
    ],
)
def test_mda_linearization(
    sellar_with_2d_array, input_data, disciplines, n, cls, options
) -> None:
    """Check the Jacobian value of the MDA."""
    mda = cls(disciplines, settings=cls.settings_class(**options))
    mda.execute(input_data)
    data = mda.input_data
    for discipline in disciplines:
        assert discipline.check_jacobian(
            data,
            derr_approx=discipline.LinearizationMode.COMPLEX_STEP,
            step=1e-30,
        )

    assert mda.check_jacobian(
        data,
        threshold=1e-4,
        derr_approx=discipline.LinearizationMode.COMPLEX_STEP,
        step=1e-30,
    )


@pytest.mark.parametrize(
    ("formulation", "differentiation_method"),
    [
        ("MDF", "user"),
        ("MDF", "complex_step"),
        ("IDF", "complex_step"),
        ("IDF", "user"),
    ],
)
def test_exec(
    sellar_with_2d_array,
    formulation,
    differentiation_method,
    disciplines,
    x_opt,
    n,
) -> None:
    """Check the resolution of the Sellar problem."""
    scenario = MDOScenario(
        disciplines,
        SellarDesignSpace(n=n),
        formulation_settings=MDO_FORMULATION_FACTORY.get_class(
            formulation
        ).settings_class(),
    )
    scenario.add_objective("obj")
    scenario.set_differentiation_method(differentiation_method)
    scenario.add_constraint(C_1, constraint_type=ArrayFunction.ConstraintType.INEQ)
    scenario.add_constraint(C_2, constraint_type=ArrayFunction.ConstraintType.INEQ)
    scenario.execute(SLSQP_Settings(max_iter=20))

    x_opt = scenario.design_space.get_current_value(as_dict=True)
    x_opt = concatenate((x_opt[X_1], x_opt[X_SHARED]))

    assert scenario.optimization_result.f_opt == pytest.approx(3.18339, rel=0.001)
    assert x_opt == pytest.approx(x_opt, abs=0.0001)


@pytest.mark.parametrize(
    ("cls", "output_names"),
    [(Sellar1, ["y_1"]), (Sellar2, ["y_2"]), (SellarSystem, ["obj", "c_1", "c_2"])],
)
@pytest.mark.parametrize("eval_jac", [False, True])
@pytest.mark.parametrize("n", [1, 2])
def test_vectorization(eval_jac, cls, output_names, n):
    """Check that the Sellar's disciplines are correctly vectorized."""
    n_samples = 3

    # Create the reference results without vectorization.
    scenario = MDOScenario([cls(n=n)], SellarDesignSpace(n=n))
    scenario.add_objective(output_names[0])
    for output_name in output_names[1:]:
        scenario.add_observable(output_name)
    scenario.execute(
        MC_Settings(n_samples=n_samples, vectorize=False, eval_jac=eval_jac)
    )
    reference = scenario.formulation.problem.database.to_dataset(export_gradients=True)

    # Create the results with vectorization.
    scenario = MDOScenario([cls(n=n)], SellarDesignSpace(n=n))
    scenario.add_objective(output_names[0])
    for output_name in output_names[1:]:
        scenario.add_observable(output_name)
    scenario.execute(
        MC_Settings(n_samples=n_samples, vectorize=True, eval_jac=eval_jac)
    )
    result = scenario.formulation.problem.database.to_dataset(export_gradients=True)

    # Compare the results
    # in terms of input values, output values and gradient values.
    assert_frame_equal(result, reference)


@pytest.mark.parametrize("eval_jac", [False, True])
@pytest.mark.parametrize("n", [1, 2])
def test_vectorization_sellar2_y_1(eval_jac, n):
    """Check that Sellar2 does not crash when y_1 is a collection of samples."""
    n_samples = 3

    input_space = DesignSpace()
    input_space.add_variable("y_1", size=n, lower_bound=0.0, upper_bound=1.0)

    # Create the reference results without vectorization.
    scenario = MDOScenario([Sellar2(n=n)], input_space)
    scenario.add_objective("y_2")
    scenario.execute(
        MC_Settings(n_samples=n_samples, vectorize=False, eval_jac=eval_jac)
    )
    reference = scenario.formulation.problem.database.to_dataset(export_gradients=True)

    input_space = DesignSpace()
    input_space.add_variable("y_1", size=n, lower_bound=0.0, upper_bound=1.0)

    # Create the results with vectorization.
    scenario = MDOScenario([Sellar2(n=n)], input_space)
    scenario.add_objective("y_2")
    scenario.execute(
        MC_Settings(n_samples=n_samples, vectorize=True, eval_jac=eval_jac)
    )
    result = scenario.formulation.problem.database.to_dataset(export_gradients=True)

    # Compare the results
    # in terms of input values, output values and gradient values.
    assert_frame_equal(result, reference)
