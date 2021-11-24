# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import pytest
from matplotlib.testing.decorators import image_comparison
from numpy import array, empty

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import create_design_space, create_discipline, create_scenario
from gemseo.core.doe_scenario import DOEScenario
from gemseo.post.post_factory import PostFactory
from gemseo.problems.sobieski.wrappers import SobieskiProblem, SobieskiStructure
from gemseo.utils.py23_compat import PY2, Path

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"


@pytest.fixture(scope="module")
def factory():
    return PostFactory()


@pytest.mark.parametrize("scale_gradients", [True, False])
def test_import_gradient_sensitivity(
    tmp_wd, factory, scale_gradients, pyplot_close_all
):
    """Test the gradient sensitivity post-processing with an imported problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
        scale_gradients: If True, normalize each gradient w.r.t. design variables.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = OptimizationProblem.import_hdf(str(POWER2))
    post = factory.execute(
        problem,
        "GradientSensitivity",
        scale_gradients=scale_gradients,
        file_path="grad_sens1",
        save=True,
    )
    assert len(post.output_files) == 1
    assert Path(post.output_files[0]).exists()

    x_0 = problem.database.get_x_by_iter(0)
    problem.database[x_0].pop("@eq")
    post = factory.execute(
        problem,
        "GradientSensitivity",
        scale_gradients=scale_gradients,
        file_path="grad_sens2",
        save=True,
        iteration=0,
    )
    assert len(post.output_files) == 1
    assert Path(post.output_files[0]).exists()


@pytest.mark.parametrize("scale_gradients", [True, False])
def test_gradient_sensitivity_prob(tmp_wd, scale_gradients, pyplot_close_all):
    """Test the gradient sensitivity post-processing with the Sobiesky problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        scale_gradients: If True, normalize each gradient w.r.t. design variables.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    disc = SobieskiStructure()
    design_space = SobieskiProblem().read_design_space()
    inputs = disc.get_input_data_names()
    design_space.filter(inputs)
    doe_scenario = DOEScenario([disc], "DisciplinaryOpt", "y_12", design_space)
    doe_scenario.execute(
        {
            "algo": "DiagonalDOE",
            "n_samples": 10,
            "algo_options": {"eval_jac": True},
        }
    )
    doe_scenario.post_process(
        "GradientSensitivity",
        scale_gradients=scale_gradients,
        file_path="grad_sens",
        save=True,
    )
    doe_scenario2 = DOEScenario([disc], "DisciplinaryOpt", "y_12", design_space)
    doe_scenario2.execute(
        {
            "algo": "DiagonalDOE",
            "n_samples": 10,
            "algo_options": {"eval_jac": False},
        }
    )

    with pytest.raises(ValueError, match="No gradients to plot at current iteration!"):
        doe_scenario2.post_process(
            "GradientSensitivity",
            file_path="grad_sens",
            save=True,
            scale_gradients=scale_gradients,
        )


# Define a simple analytical function and its Jacobian
def f(x1=0.0, x2=0.0):
    """A simple analytical test function."""
    y = 1 * x1 + 2 * x2 ** 2
    return y


def dfdxy(x1=0.0, x2=0.0):
    """Jacobian function of f."""
    jac = empty((1, 2))
    jac[0, 0] = 1
    jac[0, 1] = 4 * x2
    return jac


@pytest.mark.parametrize("scale_gradients", [True, False])
def test_scale_gradients(tmp_wd, scale_gradients, pyplot_close_all):
    """Test the analytical results of the gradient normalization.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        scale_gradients: If True, normalize each gradient w.r.t. design variables.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """

    disc = create_discipline("AutoPyDiscipline", py_func=f, py_jac=dfdxy)

    design_sp = create_design_space()
    design_sp.add_variable("x1", 1, l_b=-2.0, u_b=2.0, value=array(2.0))
    design_sp.add_variable("x2", 1, l_b=-2.0, u_b=2.0, value=array(2.0))

    scenario = create_scenario(
        disc,
        "DisciplinaryOpt",
        "y",
        design_sp,
        scenario_type="MDO",
    )
    scenario.execute(input_data={"max_iter": 10, "algo": "L-BFGS-B"})

    post = scenario.post_process(
        "GradientSensitivity",
        scale_gradients=scale_gradients,
        file_path="grad_sens_analytical",
        file_extension="png",
        save=True,
    )

    actual_jac = post._GradientSensitivity__get_grad_dict(
        array([-2.0, 0.0]), scale_gradients=scale_gradients
    )

    if scale_gradients:
        expected_jac = array([4.0, 0])
    else:
        expected_jac = array([1.0, 0.0])

    assert expected_jac.all() == actual_jac["y"].all()


@pytest.mark.skipif(PY2, reason="image comparison does not work with python 2")
@pytest.mark.parametrize(
    "scale_gradients,baseline_images",
    [(True, ["grad_sens_scaled"]), (False, ["grad_sens"])],
)
@image_comparison(None, extensions=["png"])
def test_plot(tmp_wd, baseline_images, scale_gradients, pyplot_close_all):
    """Test images created by the post_process method against references.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        baseline_images: The reference images to be compared.
        scale_gradients: If True, normalize each gradient w.r.t. design variables.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """

    disc = create_discipline("AutoPyDiscipline", py_func=f, py_jac=dfdxy)

    design_sp = create_design_space()
    design_sp.add_variable("x1", 1, l_b=-2.0, u_b=2.0, value=array(2.0))
    design_sp.add_variable("x2", 1, l_b=-2.0, u_b=2.0, value=array(2.0))

    scenario = create_scenario(
        disc,
        "DisciplinaryOpt",
        "y",
        design_sp,
        scenario_type="MDO",
    )

    scenario.execute(input_data={"max_iter": 10, "algo": "L-BFGS-B"})

    post = scenario.post_process(
        "GradientSensitivity",
        scale_gradients=scale_gradients,
        file_path="grad_sens_analytical",
        file_extension="png",
        save=False,
    )
    post.figures
