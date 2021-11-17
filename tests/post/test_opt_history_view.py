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
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import division, unicode_literals

from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import execute_algo, execute_post
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.post.opt_history_view import OptHistoryView
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.utils.py23_compat import Path

DIR_PATH = Path(__file__).parent
POWER2_PATH = DIR_PATH / "power2_opt_pb.h5"
POWER2_NAN_PATH = DIR_PATH / "power2_opt_pb_nan.h5"


def test_files_creation(tmp_wd):
    """Check the generation of the output files."""
    problem = Rosenbrock()
    OptimizersFactory().execute(problem, "L-BFGS-B")
    view = OptHistoryView(problem)
    file_path = tmp_wd / "rosen_1"
    view.execute(show=False, save=True, file_path=file_path)
    for full_path in view.output_files:
        assert Path(full_path).exists()


def test_view_load_pb(tmp_wd):
    """Check the generation of the output files from an HDF database."""
    problem = OptimizationProblem.import_hdf(POWER2_PATH)
    view = OptHistoryView(problem)
    file_path = tmp_wd / "power2view"
    view.execute(show=False, save=True, file_path=file_path, obj_relative=True)
    for full_path in view.output_files:
        assert Path(full_path).exists()


def test_view_constraints(tmp_wd):
    """Check the generation of the output files for a problem with constraints."""
    problem = Power2()
    OptimizersFactory().execute(problem, "SLSQP")
    view = OptHistoryView(problem)

    _, cstr = view._get_constraints(["toto", "ineq1"])
    assert len(cstr) == 1
    view.execute(
        show=False,
        save=True,
        variables_names=["x"],
        file_path=tmp_wd / "power2_2",
        obj_min=0.0,
        obj_max=5.0,
    )
    for full_path in view.output_files:
        assert Path(full_path).exists()


def test_nans():
    """Check the generation of the output files for a database containing NaN."""
    problem = OptimizationProblem.import_hdf(POWER2_NAN_PATH)
    view = execute_post(problem, "OptHistoryView", show=False, save=True)

    for full_path in view.output_files:
        assert Path(full_path).exists()


def test_diag_with_nan(caplog):
    """Check that the Hessian plot creation is skipped if its diagonal contains NaN."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(
        lambda x: 2 * x, "obj", jac=lambda x: array([[2.0]])
    )
    execute_algo(problem, "fullfact", n_samples=3, eval_jac=True, algo_type="doe")

    execute_post(problem, "OptHistoryView", save=False, show=False)
    log = caplog.text
    assert "Failed to create Hessian approximation." in log
    assert "ValueError: The approximated Hessian diagonal contains NaN." in log
