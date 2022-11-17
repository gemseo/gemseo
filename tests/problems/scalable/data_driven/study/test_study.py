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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import os

import pytest
from gemseo.caches.hdf5_cache import HDF5Cache
from gemseo.core.doe_scenario import DOEScenario
from gemseo.problems.scalable.data_driven.study.post import PostScalabilityStudy
from gemseo.problems.scalable.data_driven.study.process import ScalabilityStudy
from gemseo.problems.sellar.sellar import OBJ
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sellar.sellar import X_LOCAL
from gemseo.problems.sellar.sellar import X_SHARED
from gemseo.problems.sellar.sellar import Y_1
from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace


def mdf_cost(varsizes, n_c, n_lc, n_tl_c, n_tl_lc):
    cost = n_c["sdm_Sellar1"] * 1.0
    cost += n_c["sdm_Sellar2"] * 2.0
    cost += n_c["sdm_SellarSystem"] * 0.5
    return cost


def idf_cost(varsizes, n_c, n_lc, n_tl_c, n_tl_lc):
    cost = n_c["sdm_Sellar1"] * 1.0
    cost += n_c["sdm_Sellar2"] * 2.0
    cost += n_c["sdm_SellarSystem"] * 0.5
    cost += n_lc["sdm_Sellar2"] * 3 * 1.0
    cost += n_lc["sdm_Sellar2"] * 3 * 1.0
    return cost


@pytest.fixture()
def sellar_use_case(tmp_wd):
    n_samples = 20
    os.mkdir("data")
    file_name = "data/sellar.h5"
    disciplines_names = []
    for discipline_class in [Sellar1, Sellar2, SellarSystem]:
        discipline = discipline_class()
        discipline.set_cache_policy(discipline.HDF5_CACHE, cache_hdf_file=file_name)
        disciplines_names.append(discipline.name)
        objective_name = next(iter(discipline.output_grammar.keys()))
        inputs_names = list(discipline.input_grammar.keys())
        design_space = SellarDesignSpace().filter(inputs_names)
        scenario = DOEScenario(
            [discipline], "DisciplinaryOpt", objective_name, design_space
        )
        scenario.execute({"algo": "DiagonalDOE", "n_samples": n_samples})
    design_variables = [X_SHARED, X_LOCAL]
    objective_name = OBJ
    os.mkdir("study_1")
    os.mkdir("study_2")
    os.makedirs("empty_dir/results")
    return design_variables, objective_name, file_name, disciplines_names


def test_scalabilitystudy1(sellar_use_case):
    design_variables, objective, f_name, disciplines_names = sellar_use_case
    variables = [{X_SHARED: i} for i in range(1, 2)]
    directory = "study_1"
    ScalabilityStudy(
        objective,
        design_variables,
        directory,
        feasibility_level={"g_1": 0.0, "g_2": 0.0},
    )
    study = ScalabilityStudy(objective, design_variables, directory)
    for discipline_name in disciplines_names:
        study.add_discipline(HDF5Cache(f_name, discipline_name).export_to_dataset())
    assert disciplines_names == study.disciplines_names
    study.set_input_output_dependency("SellarSystem", OBJ, [Y_1])
    with pytest.raises(TypeError):
        study.set_input_output_dependency("SellarSystem", OBJ, Y_1)
    with pytest.raises(ValueError):
        study.set_input_output_dependency("SellarSystem", OBJ, ["dummy"])
    with pytest.raises(TypeError):
        study.set_input_output_dependency("SellarSystem", OBJ, [1])
    with pytest.raises(TypeError):
        study.set_input_output_dependency("SellarSystem", [OBJ], [Y_1])
    with pytest.raises(ValueError):
        study.set_input_output_dependency("SellarSystem", "dummy", [Y_1])
    with pytest.raises(TypeError):
        study.set_input_output_dependency(["SellarSystem"], OBJ, [Y_1])
    with pytest.raises(ValueError):
        study.set_input_output_dependency("dummy", OBJ, [Y_1])
    study.set_fill_factor("SellarSystem", OBJ, 0.4)
    with pytest.raises(TypeError):
        study.set_fill_factor("SellarSystem", OBJ, "high")
    with pytest.raises(TypeError):
        study.set_fill_factor("SellarSystem", OBJ, 1.4)
    with pytest.raises(ValueError):
        study.execute()
    study.add_optimization_strategy(
        "NLOPT_SLSQP", 2, "MDF", formulation_options={"chain_linearize": True}
    )
    study.add_optimization_strategy("NLOPT_SLSQP", 2, "IDF")
    study.add_scaling_strategies(variables=variables)
    study.execute()
    with pytest.raises(TypeError):
        study.add_optimization_strategy("NLOPT_SLSQP", 2, "MDF", algo_options="dummy")
    tol = 1e-4
    algo_options = {"ftol_rel": tol, "xtol_rel": tol, "ftol_abs": tol, "xtol_abs": tol}
    study.add_optimization_strategy("NLOPT_SLSQP", 2, "MDF", algo_options=algo_options)
    variables = [{X_SHARED: i} for i in range(1, 3)]

    with pytest.raises(ValueError):
        PostScalabilityStudy("dummy")
    with pytest.raises(ValueError):
        PostScalabilityStudy("")
    with pytest.raises(ValueError):
        PostScalabilityStudy("results")
    with pytest.raises(ValueError):
        PostScalabilityStudy("")

    post = PostScalabilityStudy("study_1")
    post.labelize_exec_time("exec_time")
    post.labelize_n_calls("n_calls")
    post.labelize_n_calls_linearize("n_calls_linearize")
    post.labelize_status("status")
    post.labelize_is_feasible("is_feasible")
    post.labelize_scaling_strategy("scaling_strategy")
    post.set_cost_unit("h")
    post.labelize_original_exec_time("original_exec_time")
    with pytest.raises(TypeError):
        post.labelize_original_exec_time(123)
    with pytest.raises(ValueError):
        post._update_descriptions("dummy", "description")

    post.set_cost_function("MDF", mdf_cost)
    post.set_cost_function("IDF", mdf_cost)
    post.plot()
    post.get_scaling_strategies(True)


def test_scalabilitystudy2(sellar_use_case):
    design_variables, objective, f_name, disciplines_names = sellar_use_case
    variables = [{X_SHARED: i} for i in range(1, 3)]
    study = ScalabilityStudy(objective, design_variables, "study_2")
    for discipline_name in disciplines_names:
        study.add_discipline(HDF5Cache(f_name, discipline_name).export_to_dataset())
    study.add_optimization_strategy("NLOPT_SLSQP", 2, "MDF")
    study.add_optimization_strategy("NLOPT_SLSQP", 2, "IDF")
    study.add_scaling_strategies(
        coupling_size=1, eq_cstr_size=[1, 2], variables=variables
    )
    study.execute(2)

    post = PostScalabilityStudy("study_2")
    post.set_cost_function("MDF", mdf_cost)
    with pytest.raises(ValueError):
        post._estimate_original_time()
    post.set_cost_function("IDF", mdf_cost)
    post.plot()
    post.plot(widths=[0.25, 0.25], xticks=[1, 2])

    assert post.n_samples == 2

    with pytest.raises(ValueError):
        post = PostScalabilityStudy("Dummy")
