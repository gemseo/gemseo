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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import division, unicode_literals

import pytest
from numpy import array, ones, zeros
from scipy.optimize import rosen, rosen_der

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import create_design_space, create_mda, create_scenario, execute_algo
from gemseo.core.auto_py_discipline import AutoPyDiscipline, to_arrays_dict
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.py23_compat import Path


def create_ds(n):
    design_space = create_design_space()
    design_space.add_variable("x", n, l_b=-2 * ones(n), u_b=2 * ones(n), value=zeros(n))
    return design_space


def f1(y2=1.0, z=2.0):
    y1 = z + y2
    return y1


def f2(y1=2.0, z=2.0):
    y2 = z + 2 * y1
    y3 = 14
    return y2, y3


def f3(x=1.0):
    if x > 0:
        y = -x
        return y
    y = 2 * x
    return y


def f4(x=1.0):
    if x > 0:
        y = -x
        return y
    y = 2 * x
    return y, x


def test_basic():
    """Test a basic auto-discipline execution."""
    d1 = AutoPyDiscipline(f1)
    d1.execute()

    assert d1.local_data["y1"] == f1()

    d2 = AutoPyDiscipline(f2)
    d2.execute()
    assert d2.local_data["y2"] == f2()[0]


def test_write_schema(tmp_wd):
    """Test the writing of the schema."""
    d1 = AutoPyDiscipline(f1, write_schema=True)
    d1.execute()
    for trailer in ["input", "output"]:
        path = Path("f1_{}.json".format(trailer))
        assert path.is_file()


def test_use_arrays():
    """Test the use of arrays."""
    d1 = AutoPyDiscipline(f1, use_arrays=True)
    d1.execute()
    assert d1.local_data["y1"] == f1()
    d1.execute({"x1": array([1.0]), "z": array([2.0])})
    assert d1.local_data["y1"] == f1()


def test_mda():
    """Test a MDA of AutoPyDisciplines."""
    d1 = AutoPyDiscipline(f1)
    d2 = AutoPyDiscipline(f2)
    mda = create_mda("MDAGaussSeidel", [d1, d2])
    mda.execute()


def test_fail_wrongly_formatted_function():
    """Test that a wrongly formatted function cannot be used."""
    AutoPyDiscipline(f3)
    with pytest.raises(ValueError):
        AutoPyDiscipline(f4)


def test_fail_not_a_python_function():
    """Test the failure if a Python function is not provided."""
    not_a_function = 2
    with pytest.raises(TypeError):
        AutoPyDiscipline(not_a_function)


def test_jac_pb():
    """Test the AutoPyDiscipline with Jacobian provided."""
    max_iter = 100
    n = 4
    algo = "L-BFGS-B"

    design_space = create_ds(n)
    pb = OptimizationProblem(design_space)
    pb.objective = MDOFunction(rosen, "rosen", jac=rosen_der)
    execute_algo(pb, algo, max_iter=max_iter)
    fopt_ref = pb.solution.f_opt

    design_space = create_ds(n)
    auto_rosen = AutoPyDiscipline(rosen, rosen_der)
    scn = create_scenario(auto_rosen, "DisciplinaryOpt", "r", design_space)
    scn.execute({"algo": algo, "max_iter": max_iter})
    scn_opt = scn.optimization_result.f_opt

    assert fopt_ref == scn_opt

    auto_rosen = AutoPyDiscipline(rosen)
    with pytest.raises(RuntimeError):
        auto_rosen._compute_jacobian()


@pytest.mark.parametrize("input", [{"a": [1.0]}, {"a": array([1.0])}])
def test_to_arrays_dict(input):
    """Test the function to_arrays_dict."""
    output = to_arrays_dict(input)
    assert output["a"] == array([1.0])
