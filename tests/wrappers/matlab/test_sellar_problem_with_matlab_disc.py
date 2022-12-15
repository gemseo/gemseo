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

import numpy as np
import pytest

# skip if matlab API is not found
pytest.importorskip("matlab")

from gemseo.algos.design_space import DesignSpace  # noqa: E402
from gemseo.api import create_discipline, create_scenario  # noqa: E402

from .matlab_files import MATLAB_FILES_DIR_PATH  # noqa: E402


def build_matlab_disciplines():
    """Build all matlab discipline for Sellar problem.

    Jacobian matrices are returned by matlab functions.
    """
    matlab_data = MATLAB_FILES_DIR_PATH / "sellar_data.mat"

    sellar1 = create_discipline(
        "MatlabDiscipline",
        matlab_fct="Sellar1.m",
        matlab_data_file=matlab_data,
        name="sellar_1",
        search_file=MATLAB_FILES_DIR_PATH,
        is_jac_returned_by_func=True,
    )

    sellar2 = create_discipline(
        "MatlabDiscipline",
        matlab_fct="Sellar2.m",
        matlab_data_file=matlab_data,
        search_file=MATLAB_FILES_DIR_PATH,
        name="sellar_2",
        is_jac_returned_by_func=True,
    )

    sellar_system = create_discipline(
        "MatlabDiscipline",
        matlab_fct="SellarSystem.m",
        matlab_data_file=matlab_data,
        search_file=MATLAB_FILES_DIR_PATH,
        name="sellar_system",
        is_jac_returned_by_func=True,
    )

    return [sellar1, sellar2, sellar_system]


def build_matlab_scenario():
    """Build the Sellar scenario for matlab tests."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=10.0, value=np.ones(1))
    design_space.add_variable(
        "z", 2, l_b=(-10, 0.0), u_b=(10.0, 10.0), value=np.array([4.0, 3.0])
    )
    design_space.add_variable("y_1", l_b=-100.0, u_b=100.0, value=np.ones(1))
    design_space.add_variable("y_2", l_b=-100.0, u_b=100.0, value=np.ones(1))

    disciplines = build_matlab_disciplines()
    scenario = create_scenario(
        disciplines, formulation="IDF", objective_name="obj", design_space=design_space
    )

    scenario.add_constraint("c_1", "ineq")
    scenario.add_constraint("c_2", "ineq")

    return scenario


def test_matlab_jacobians_sellar1():
    """Check that jacobian matrices returned by matlab functions are correct with respect
    to finite difference computation for Sellar1."""
    sellar1, sellar2, sellar_system = build_matlab_disciplines()

    threshold = 1e-7
    step = 1e-7

    assert sellar1.check_jacobian(step=step, threshold=threshold)
    assert sellar2.check_jacobian(step=step, threshold=threshold)
    assert sellar_system.check_jacobian(step=step, threshold=threshold)


def test_matlab_optim_results():
    """Test obtained optimal values when solving sellar problem with matlab discipline.

    Jacobians are computed.
    """
    scenario = build_matlab_scenario()
    scenario.execute(input_data={"max_iter": 20, "algo": "SLSQP"})

    # ref values are taken from the doc "Sellar Problem"

    optim_res = scenario.get_optimum()
    assert pytest.approx(optim_res.f_opt) == 3.182059

    x_opt = scenario.design_space.get_current_value(as_dict=True)
    assert pytest.approx(x_opt["x"]) == 0.0
    assert pytest.approx(x_opt["z"][0]) == 2.0363869
    assert pytest.approx(x_opt["z"][1]) == 0.0
    assert pytest.approx(x_opt["y_1"]) == 3.16
    assert pytest.approx(x_opt["y_2"]) == 3.814028
