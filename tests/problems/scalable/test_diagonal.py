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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import absolute_import, division, unicode_literals

from os.path import exists, join

import pytest
from future import standard_library

from gemseo.algos.design_space import DesignSpace
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.problems.scalable.diagonal import ScalableDiagonalModel

standard_library.install_aliases()


@pytest.fixture
def dataset():
    expressions_dict = {
        "y1": "sin(2*pi*x1)+cos(2*pi*x2)+x3",
        "y2": "sin(2*pi*x1)*cos(2*pi*x2)-x3",
    }
    disc = AnalyticDiscipline(name="sinus", expressions_dict=expressions_dict)
    disc.set_cache_policy(disc.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x1", l_b=0, u_b=1)
    design_space.add_variable("x2", l_b=0, u_b=1)
    design_space.add_variable("x3", l_b=0, u_b=1)
    doe = DOEScenario([disc], "DisciplinaryOpt", "y1", design_space)
    doe.execute({"algo": "fullfact", "n_samples": 10})
    return disc.cache


def test_constructor(dataset):
    model = ScalableDiagonalModel(dataset)
    with pytest.raises(TypeError):
        ScalableDiagonalModel(dataset, fill_factor="dummy")


def test_scalable_function(dataset):
    model = ScalableDiagonalModel(dataset)
    output = model.scalable_function()
    assert "y1" in output
    assert "y2" in output
    assert len(output["y1"].shape) == 1
    assert len(output["y1"] == 3)
    assert len(output["y2"].shape) == 1
    assert len(output["y2"] == 3)


def test_scalable_derivative(dataset):
    model = ScalableDiagonalModel(dataset)
    output = model.scalable_derivatives()
    assert "y1" in output
    assert "y2" in output
    assert len(output["y1"].shape) == 2
    assert output["y1"].shape[0] == 1
    assert output["y1"].shape[1] == 3
    assert len(output["y2"].shape) == 2
    assert output["y2"].shape[0] == 1
    assert output["y2"].shape[1] == 3


def test_plot(dataset, tmp_path):
    model = ScalableDiagonalModel(dataset)
    model.plot_1d_interpolations(save=True, show=False, directory=str(tmp_path))
    assert exists(join(str(tmp_path), "sdm_sinus_y1_1D_interpolation_0.pdf"))
    assert exists(join(str(tmp_path), "sdm_sinus_y2_1D_interpolation_0.pdf"))
    model.plot_dependency(save=True, show=False, directory=str(tmp_path))
    assert exists(join(str(tmp_path), "sdm_sinus_dependency.pdf"))


def test_force_io_dependency(dataset):
    model = ScalableDiagonalModel(dataset, force_input_dependency=True)


def test_force_allow_unusedinputs(dataset):
    model = ScalableDiagonalModel(dataset, allow_unused_inputs=False)


def test_wrong_fill_factor(dataset):
    with pytest.raises(TypeError):
        ScalableDiagonalModel(dataset, fill_factor=-2)


def test_group_dep(dataset):
    model = ScalableDiagonalModel(dataset, group_dep={"y2": ["x1", "x2"]})
    assert model.model.io_dependency["y2"]["x3"][0] == 0.0
