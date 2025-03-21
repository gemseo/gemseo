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
import re
from pathlib import Path
from unittest import mock

import pytest
from numpy import array
from numpy import empty

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post.factory import PostFactory
from gemseo.post.gradient_sensitivity import GradientSensitivity
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.utils.testing.helpers import image_comparison

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"
SOBIESKI_MISSING_GRADIENTS = Path(__file__).parent / "sobieski_missing_gradients.h5"
SOBIESKI_ALL_GRADIENTS = Path(__file__).parent / "sobieski_all_gradients.h5"
SOBIESKI_GRADIENT_VALUES = Path(__file__).parent / "sobieski_gradient.pkl"


@pytest.fixture(scope="module")
def factory():
    return PostFactory()


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
        post_name="GradientSensitivity",
        scale_gradients=scale_gradients,
        file_path="grad_sens1",
        save=True,
    )
    assert len(post.output_file_paths) == 1
    assert Path(post.output_file_paths[0]).exists()

    x_0 = problem.database.get_x_vect(1)
    problem.database[x_0].pop("@eq")
    post = factory.execute(
        problem,
        post_name="GradientSensitivity",
        scale_gradients=scale_gradients,
        file_path="grad_sens2",
        save=True,
        iteration=1,
    )
    assert len(post.output_file_paths) == 1
    assert Path(post.output_file_paths[0]).exists()


@pytest.mark.parametrize("scale_gradients", [True, False])
def test_gradient_sensitivity_prob(tmp_wd, scale_gradients) -> None:
    """Test the gradient sensitivity post-processing with the Sobieski problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        scale_gradients: If True, normalize each gradient w.r.t. design variables.
    """
    disc = SobieskiStructure()
    design_space = SobieskiDesignSpace()
    inputs = [name for name in disc.io.input_grammar if not name.startswith("c_")]
    design_space.filter(inputs)
    doe_scenario = DOEScenario(
        [disc], "y_12", design_space, formulation_name="DisciplinaryOpt"
    )
    doe_scenario.execute(algo_name="DiagonalDOE", n_samples=10, eval_jac=True)
    doe_scenario.post_process(
        post_name="GradientSensitivity",
        scale_gradients=scale_gradients,
        file_path="grad_sens",
        save=True,
    )
    doe_scenario2 = DOEScenario(
        [disc], "y_12", design_space, formulation_name="DisciplinaryOpt"
    )
    doe_scenario2.execute(algo_name="DiagonalDOE", n_samples=10, eval_jac=False)

    with pytest.raises(
        ValueError, match=re.escape("No gradients to plot at current iteration.")
    ):
        doe_scenario2.post_process(
            post_name="GradientSensitivity",
            file_path="grad_sens",
            save=True,
            scale_gradients=scale_gradients,
        )


# Define a simple analytical function and its Jacobian
def f(x1=0.0, x2=0.0):
    """A simple analytical test function."""
    y = 1 * x1 + 2 * x2**2
    return y  # noqa: RET504


def dfdxy(x1=0.0, x2=0.0):
    """Jacobian function of f."""
    jac = empty((1, 2))
    jac[0, 0] = 1
    jac[0, 1] = 4 * x2
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
    scenario.execute(algo_name="L-BFGS-B", max_iter=10)

    post = scenario.post_process(
        post_name="GradientSensitivity",
        scale_gradients=scale_gradients,
        file_path="grad_sens_analytical",
        file_extension="png",
        save=True,
    )

    actual_jac = post._GradientSensitivity__get_output_gradients(
        array([-2.0, 0.0]), scale_gradients=scale_gradients
    )

    expected_jac = array([4.0, 0]) if scale_gradients else array([1.0, 0.0])

    assert expected_jac.all() == actual_jac["y"].all()


@pytest.mark.parametrize(
    ("scale_gradients", "baseline_images"),
    [(True, ["grad_sens_scaled"]), (False, ["grad_sens"])],
)
@image_comparison(None)
def test_plot(tmp_wd, baseline_images, scale_gradients) -> None:
    """Test images created by the post_process method against references.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        baseline_images: The reference images to be compared.
        scale_gradients: If True, normalize each gradient w.r.t. design variables.
    """
    disc = create_discipline("AutoPyDiscipline", py_func=f, py_jac=dfdxy)

    design_sp = create_design_space()
    design_sp.add_variable("x1", lower_bound=-2.0, upper_bound=2.0, value=array(2.0))
    design_sp.add_variable("x2", lower_bound=-2.0, upper_bound=2.0, value=array(2.0))

    scenario = create_scenario(disc, "y", design_sp, formulation_name="DisciplinaryOpt")

    scenario.execute(algo_name="L-BFGS-B", max_iter=10)

    scenario.post_process(
        post_name="GradientSensitivity",
        scale_gradients=scale_gradients,
        file_path="grad_sens_analytical",
        file_extension="png",
        save=False,
    )


TEST_PARAMETERS = {
    "standardized": (True, ["GradientSensitivity_standardized"]),
    "unstandardized": (False, ["GradientSensitivity_unstandardized"]),
}


@pytest.mark.parametrize(
    ("use_standardized_objective", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(
    use_standardized_objective, baseline_images, common_problem
) -> None:
    """Check GradientSensitivity with objective, standardized or not."""
    opt = GradientSensitivity(common_problem)
    common_problem.use_standardized_objective = use_standardized_objective
    opt.execute(save=False)


@pytest.mark.parametrize(
    ("compute_missing_gradients", "opt_problem", "baseline_images"),
    [
        (True, SOBIESKI_ALL_GRADIENTS, ["grad_sens_sobieski"]),
        (
            True,
            SOBIESKI_MISSING_GRADIENTS,
            [],
        ),
        (False, SOBIESKI_ALL_GRADIENTS, ["grad_sens_sobieski"]),
        (False, SOBIESKI_MISSING_GRADIENTS, []),
    ],
)
@image_comparison(None)
def test_compute_missing_gradients(
    compute_missing_gradients,
    opt_problem,
    baseline_images,
    factory,
    caplog,
) -> None:
    """Test the option to compute missing gradients for a given iteration.

    Args:
        compute_missing_gradients: Whether to compute the gradients if they are missing.
        opt_problem: The path to an HDF5 file of the Sobieski problem.
        baseline_images: The references images for the image comparison test.
        factory: Fixture that returns a post-processing factory.
        caplog: Fixture to access and control log capturing.
    """
    problem = OptimizationProblem.from_hdf(str(opt_problem))

    if opt_problem == SOBIESKI_MISSING_GRADIENTS:
        with pytest.raises(
            ValueError, match=re.escape("No gradients to plot at current iteration.")
        ):
            factory.execute(
                problem,
                post_name="GradientSensitivity",
                compute_missing_gradients=compute_missing_gradients,
                save=False,
                show=False,
            )

        if compute_missing_gradients:
            assert (
                "The missing gradients for an OptimizationProblem without callable "
                "functions cannot be computed." in caplog.text
            )
    else:
        factory.execute(
            problem,
            post_name="GradientSensitivity",
            compute_missing_gradients=compute_missing_gradients,
            save=False,
            show=False,
        )


@image_comparison(["grad_sens_sobieski"])
def test_compute_missing_gradients_with_eval(factory) -> None:
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
            post_name="GradientSensitivity",
            compute_missing_gradients=True,
            save=False,
            show=False,
        )
        mocked_evaluate_functions.assert_called()
