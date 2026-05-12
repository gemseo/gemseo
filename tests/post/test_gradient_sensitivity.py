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
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pickle
from pathlib import Path
from unittest import mock

import pytest
from numpy import array
from numpy import empty

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.doe.diagonal_doe.settings.diagonal_doe_settings import (
    DiagonalDOE_Settings,
)
from gemseo.algos.opt.scipy_local.settings.lbfgsb import L_BFGS_B_Settings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post import GradientSensitivity_Settings
from gemseo.post.factory import POST_FACTORY
from gemseo.post.gradient_sensitivity import GradientSensitivity
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo import MDOScenario
from gemseo.utils.testing.helpers import assert_exception

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"
SOBIESKI_MISSING_GRADIENTS = Path(__file__).parent / "sobieski_missing_gradients.h5"
SOBIESKI_ALL_GRADIENTS = Path(__file__).parent / "sobieski_all_gradients.h5"
SOBIESKI_GRADIENT_VALUES = Path(__file__).parent / "sobieski_gradient.pkl"


@pytest.fixture(scope="module")
def factory():
    return POST_FACTORY


@pytest.mark.parametrize("scale_gradients", [True, False])
def test_import_gradient_sensitivity(tmp_wd, factory, scale_gradients) -> None:
    """Test the gradient sensitivity post-processing with an imported problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
        scale_gradients: If True, normalize each gradient w.r.t. design variables.
    """
    problem = OptimizationProblem.from_hdf(POWER2)
    post = factory.execute(
        problem,
        GradientSensitivity_Settings(
            scale_gradients=scale_gradients, file_path="grad_sens1", save=True
        ),
    )
    assert len(post.output_file_paths) == 1
    assert Path(post.output_file_paths[0]).exists()

    x_0 = problem.database.get_x_vect(1)
    problem.database[x_0].pop("@eq")
    post = factory.execute(
        problem,
        GradientSensitivity_Settings(
            scale_gradients=scale_gradients,
            file_path="grad_sens2",
            save=True,
            iteration=1,
        ),
    )
    assert len(post.output_file_paths) == 1
    assert Path(post.output_file_paths[0]).exists()


@pytest.mark.parametrize("scale_gradients", [True, False])
def test_gradient_sensitivity_prob(tmp_wd, scale_gradients, snapshot) -> None:
    """Test the gradient sensitivity post-processing with the Sobieski problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        scale_gradients: If True, normalize each gradient w.r.t. design variables.
    """
    disc = SobieskiStructure()
    design_space = SobieskiDesignSpace()
    inputs = [name for name in disc.io.input_grammar if not name.startswith("c_")]
    design_space.filter(inputs)
    mdo_scenario = MDOScenario([disc], design_space)
    mdo_scenario.add_objective("y_12")
    mdo_scenario.execute(DiagonalDOE_Settings(n_samples=10, eval_jac=True))
    mdo_scenario.post_process(
        GradientSensitivity_Settings(
            scale_gradients=scale_gradients, file_path="grad_sens", save=True
        )
    )
    mdo_scenario2 = MDOScenario([disc], design_space)
    mdo_scenario2.add_objective("y_12")
    mdo_scenario2.execute(DiagonalDOE_Settings(n_samples=10, eval_jac=False))

    with assert_exception(ValueError, snapshot):
        mdo_scenario2.post_process(
            GradientSensitivity_Settings(
                file_path="grad_sens", save=True, scale_gradients=scale_gradients
            )
        )


# Define a simple analytical function and its Jacobian
def f(x1=0.0, x2=0.0):
    """A simple analytical test function."""
    y = 1 * x1 + 2 * x2**2
    return y  # noqa: RET504


def dfdxy(x1, x2):
    """Jacobian function of f."""
    jac = empty((1, 2))
    jac[0, 0] = 1
    jac[0, 1] = 4 * x2[0]
    return jac


@pytest.mark.parametrize("scale_gradients", [True, False])
def test_scale_gradients(tmp_wd, scale_gradients) -> None:
    """Test the analytical results of the gradient normalization.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        scale_gradients: If True, normalize each gradient w.r.t. design variables.
    """
    disc = create_discipline("AutoPyDiscipline", py_func=f, py_jac=dfdxy)

    design_sp = create_design_space()
    design_sp.add_variable("x1", lower_bound=-2.0, upper_bound=2.0, value=array(2.0))
    design_sp.add_variable("x2", lower_bound=-2.0, upper_bound=2.0, value=array(2.0))

    scenario = create_scenario(disc, "y", design_sp, formulation_name="DisciplinaryOpt")
    scenario.execute(L_BFGS_B_Settings(max_iter=10))

    post = scenario.post_process(
        GradientSensitivity_Settings(
            scale_gradients=scale_gradients,
            file_path="grad_sens_analytical",
            file_extension="png",
            save=True,
        )
    )

    actual_jac = post._GradientSensitivity__get_output_gradients(
        array([-2.0, 0.0]), scale_gradients=scale_gradients
    )

    expected_jac = array([4.0, 0]) if scale_gradients else array([1.0, 0.0])

    assert expected_jac.all() == actual_jac["y"].all()


@pytest.mark.parametrize(
    "scale_gradients",
    [True, False],
)
def test_plot(scale_gradients, snapshot_matplotlib) -> None:
    """Test images created by the post_process method against references.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        scale_gradients: If True, normalize each gradient w.r.t. design variables.
    """
    disc = create_discipline("AutoPyDiscipline", py_func=f, py_jac=dfdxy)

    design_sp = create_design_space()
    design_sp.add_variable("x1", lower_bound=-2.0, upper_bound=2.0, value=array(2.0))
    design_sp.add_variable("x2", lower_bound=-2.0, upper_bound=2.0, value=array(2.0))

    scenario = create_scenario(disc, "y", design_sp, formulation_name="DisciplinaryOpt")

    scenario.execute(L_BFGS_B_Settings(max_iter=10))

    scenario.post_process(
        GradientSensitivity_Settings(
            scale_gradients=scale_gradients,
            file_path="grad_sens_analytical",
            file_extension="png",
            save=False,
        )
    )


@pytest.mark.parametrize(
    "use_standardized_objective",
    [True, False],
)
def test_common_scenario(
    use_standardized_objective, common_problem, snapshot_matplotlib
) -> None:
    """Check GradientSensitivity with objective, standardized or not."""
    common_problem.use_standardized_objective = use_standardized_objective
    opt = GradientSensitivity(common_problem)
    opt.execute(GradientSensitivity_Settings(save=False))


@pytest.mark.parametrize(
    ("compute_missing_gradients", "opt_problem"),
    [
        (True, SOBIESKI_ALL_GRADIENTS),
        (True, SOBIESKI_MISSING_GRADIENTS),
        (False, SOBIESKI_ALL_GRADIENTS),
        (False, SOBIESKI_MISSING_GRADIENTS),
    ],
)
def test_compute_missing_gradients(
    compute_missing_gradients,
    opt_problem,
    factory,
    caplog,
    snapshot,
) -> None:
    """Test the option to compute missing gradients for a given iteration.

    Args:
        compute_missing_gradients: Whether to compute the gradients if they are missing.
        opt_problem: The path to an HDF5 file of the Sobieski problem.
        factory: Fixture that returns a post-processing factory.
        caplog: Fixture to access and control log capturing.
    """
    problem = OptimizationProblem.from_hdf(str(opt_problem))

    if opt_problem == SOBIESKI_MISSING_GRADIENTS:
        with assert_exception(ValueError, snapshot):
            factory.execute(
                problem,
                GradientSensitivity_Settings(
                    compute_missing_gradients=compute_missing_gradients,
                    save=False,
                    show=False,
                ),
            )

        if compute_missing_gradients:
            assert (
                "The missing gradients for an OptimizationProblem without callable "
                "functions cannot be computed." in caplog.text
            )
    else:
        factory.execute(
            problem,
            GradientSensitivity_Settings(
                compute_missing_gradients=compute_missing_gradients,
                save=False,
                show=False,
            ),
        )


def test_compute_missing_gradients_with_eval(factory, snapshot_matplotlib) -> None:
    """Test that the computation of the missing gradients works well with functions.

    Args:
        factory: Fixture that returns a post-processing factory.
    """
    problem = OptimizationProblem.from_hdf(str(SOBIESKI_MISSING_GRADIENTS))

    with open(SOBIESKI_GRADIENT_VALUES, "rb") as handle:
        gradients = pickle.load(handle)

    with mock.patch.object(problem, "evaluate_functions") as mocked_evaluate_functions:
        mocked_evaluate_functions.return_value = (None, gradients)
        factory.execute(
            problem,
            GradientSensitivity_Settings(
                compute_missing_gradients=True, save=False, show=False
            ),
        )
        mocked_evaluate_functions.assert_called()
