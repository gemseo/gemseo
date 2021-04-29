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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, unicode_literals

import pytest
from numpy import allclose

from gemseo.algos.design_space import DesignSpace
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.surrogate_disc import SurrogateDiscipline
from gemseo.mlearning.regression.linreg import LinearRegression

LEARNING_SIZE = 9


@pytest.fixture
def dataset():
    """Dataset from a R^2 -> R^2 function sampled over [0,1]^2."""
    expressions_dict = {"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"}
    discipline = AnalyticDiscipline("func", expressions_dict)
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0)
    design_space.add_variable("x_2", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": LEARNING_SIZE})
    data = discipline.cache.export_to_dataset()
    return data


def test_constructor(dataset):
    with pytest.raises(ValueError):
        SurrogateDiscipline("LinearRegression")
    surr = SurrogateDiscipline("LinearRegression", dataset)
    assert surr.linearization_mode == "auto"
    surr = SurrogateDiscipline("GaussianProcessRegression", dataset)
    assert surr.linearization_mode == "finite_differences"
    assert set(["x_1", "x_2"]) == set(surr.get_input_data_names())
    assert set(["y_1", "y_2"]) == set(surr.get_output_data_names())


def test_constructor_from_algo(dataset):
    algo = LinearRegression(dataset)
    algo.learn()
    surr = SurrogateDiscipline(algo)
    assert surr.linearization_mode == "auto"
    assert set(["x_1", "x_2"]) == set(surr.get_input_data_names())
    assert set(["y_1", "y_2"]) == set(surr.get_output_data_names())


def test_repr(dataset):
    surr = SurrogateDiscipline("LinearRegression", dataset)
    msg = "SurrogateDiscipline(name=LinReg_func, "
    msg += "algo=LinearRegression, data=func, "
    msg += "size=9, inputs=[x_1, x_2], outputs=[y_1, y_2], jacobian=auto)"
    assert repr(surr) == msg


def test_str(dataset):
    surr = SurrogateDiscipline("LinearRegression", dataset)
    msg = "Surrogate discipline: LinReg_func\n"
    msg += "   Dataset name: func\n"
    msg += "   Dataset size: 9\n"
    msg += "   Surrogate model: LinearRegression\n"
    msg += "   Inputs: x_1, x_2\n"
    msg += "   Outputs: y_1, y_2"
    assert str(surr) == msg


def test_execute(dataset):
    surr = SurrogateDiscipline("LinearRegression", dataset)
    out = surr.execute()
    assert "y_1" in out
    assert "y_2" in out
    assert allclose(out["y_1"][0], 3.5, atol=1e-3)
    assert allclose(out["y_2"][0], -3.5, atol=1e-3)


def test_linearize(dataset):
    surr = SurrogateDiscipline("LinearRegression", dataset)
    out = surr.linearize()
    assert "y_1" in out
    assert "y_2" in out
    assert "x_1" in out["y_1"]
    assert "x_2" in out["y_1"]
    assert "x_1" in out["y_2"]
    assert "x_2" in out["y_2"]
    assert allclose(out["y_1"]["x_1"][0], 2.0)
    assert allclose(out["y_1"]["x_2"][0], 3.0)
    assert allclose(out["y_2"]["x_1"][0], -2.0)
    assert allclose(out["y_2"]["x_2"][0], -3.0)
