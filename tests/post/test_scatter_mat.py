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

from os.path import dirname, exists, join

import pytest
from matplotlib.testing.decorators import image_comparison
from numpy import array, ones

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import create_design_space, create_discipline, create_scenario
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.utils.py23_compat import PY2

POWER2 = join(dirname(__file__), "power2_opt_pb.h5")


def test_scatter(tmp_wd, pyplot_close_all):
    """Test the scatter matrix post-processing for all functions.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    factory = PostFactory()
    if factory.is_available("ScatterPlotMatrix"):
        problem = Power2()
        OptimizersFactory().execute(problem, "SLSQP")
        post = factory.execute(
            problem,
            "ScatterPlotMatrix",
            save=True,
            file_path="scatter1",
            variables_list=problem.get_all_functions_names(),
        )
        assert len(post.output_files) == 1
        for outf in post.output_files:
            assert exists(outf)


def test_scatter_load(tmp_wd, pyplot_close_all):
    """Test scatter matrix post-processing with an imported problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    factory = PostFactory()
    if factory.is_available("ScatterPlotMatrix"):
        problem = OptimizationProblem.import_hdf(POWER2)
        post = factory.execute(
            problem,
            "ScatterPlotMatrix",
            save=True,
            file_path="scatter2",
            variables_list=problem.get_all_functions_names(),
        )
        assert len(post.output_files) == 1
        for outf in post.output_files:
            assert exists(outf)

        with pytest.raises(Exception):
            factory.execute(
                problem, "ScatterPlotMatrix", save=True, variables_list=["I dont exist"]
            )

        post = factory.execute(
            problem, "ScatterPlotMatrix", save=True, variables_list=[]
        )
        for outf in post.output_files:
            assert exists(outf)


@pytest.mark.skipif(PY2, reason="image comparison does not work with python 2")
@pytest.mark.parametrize(
    "variables_list, baseline_images",
    [
        ([], ["empty_list"]),
        (["x_shared", "obj"], ["subset_2components"]),
        (["x_shared", "x_local"], ["subset_2variables"]),
        (["c_2", "x_shared", "x_local", "obj", "c_1"], ["all_var_func"]),
    ],
)
@image_comparison(None, extensions=["png"])
def test_scatter_plot(tmp_wd, baseline_images, variables_list, pyplot_close_all):
    """Test images created by the post_process method against references.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        baseline_images (list): The reference images to be compared.
        variables_list (list): The list of variables to be plotted
            in each test case.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
    design_space = create_design_space()
    design_space.add_variable("x_local", 1, l_b=0.0, u_b=10.0, value=ones(1))
    design_space.add_variable(
        "x_shared", 2, l_b=(-10, 0.0), u_b=(10.0, 10.0), value=array([4.0, 3.0])
    )
    design_space.add_variable("y_0", 1, l_b=-100.0, u_b=100.0, value=ones(1))
    design_space.add_variable("y_1", 1, l_b=-100.0, u_b=100.0, value=ones(1))
    scenario = create_scenario(
        disciplines, "MDF", objective_name="obj", design_space=design_space
    )
    scenario.add_constraint("c_1", "ineq")
    scenario.add_constraint("c_2", "ineq")
    scenario.set_differentiation_method("finite_differences", 1e-6)
    scenario.default_inputs = {"max_iter": 10, "algo": "SLSQP"}
    scenario.execute()
    post = scenario.post_process(
        "ScatterPlotMatrix",
        save=False,
        file_path="scatter_sellar",
        file_extension="png",
        variables_list=variables_list,
    )
    post.figures


def test_maximized_func(tmp_wd, pyplot_close_all):
    """Test if the method identifies maximized objectives properly.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
    design_space = create_design_space()
    design_space.add_variable("x_local", 1, l_b=0.0, u_b=10.0, value=ones(1))
    design_space.add_variable(
        "x_shared", 2, l_b=(-10, 0.0), u_b=(10.0, 10.0), value=array([4.0, 3.0])
    )
    design_space.add_variable("y_0", 1, l_b=-100.0, u_b=100.0, value=ones(1))
    design_space.add_variable("y_1", 1, l_b=-100.0, u_b=100.0, value=ones(1))
    scenario = create_scenario(
        disciplines,
        "MDF",
        objective_name="obj",
        design_space=design_space,
        maximize_objective=True,
    )
    scenario.add_constraint("c_1", "ineq")
    scenario.add_constraint("c_2", "ineq")
    scenario.set_differentiation_method("finite_differences", 1e-6)
    scenario.default_inputs = {"max_iter": 10, "algo": "SLSQP"}
    scenario.execute()
    post = scenario.post_process(
        "ScatterPlotMatrix",
        save=True,
        file_path="scatter_sellar",
        file_extension="png",
        variables_list=["obj", "x_local", "x_shared"],
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert exists(outf)
