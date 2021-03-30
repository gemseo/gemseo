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
from future import standard_library
from numpy import allclose, array, linspace, savetxt

from gemseo.uncertainty.statistics.empirical import EmpiricalStatistics

standard_library.install_aliases()


@pytest.fixture(scope="module")
def dataset():
    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.analytic_discipline import AnalyticDiscipline
    from gemseo.core.doe_scenario import DOEScenario

    discipline = AnalyticDiscipline(
        expressions_dict={"obj": "x_1+x_2", "cstr": "x_1-x_2"}
    )
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", 1, "float", 1.0, 10.0, 5.0)
    design_space.add_variable("x_2", 1, "float", 2.0, 11.0, 5.0)
    design_space.filter(discipline.get_input_data_names())
    scenario = DOEScenario([discipline], "MDF", "obj", design_space)
    scenario.add_constraint("cstr")
    scenario.execute({"algo": "DiagonalDOE", "n_samples": 10})
    return discipline.cache.export_to_dataset()


@pytest.fixture(scope="module")
def mc_datasets(dataset):
    stats = EmpiricalStatistics(dataset)
    stats_obj = EmpiricalStatistics(dataset, ["obj"])
    return (stats, stats_obj, dataset)


def test_empstats_properties(mc_datasets):
    stats, _, dataset = mc_datasets
    assert stats.n_samples == dataset.length
    assert stats.n_variables == dataset.n_variables
    assert stats.names == dataset.variables


def test_empstats_minmax(mc_datasets):
    stats, _, _ = mc_datasets
    tmp = stats.minimum()
    assert allclose(tmp["cstr"][0], -1.0)
    assert allclose(tmp["obj"][0], 3.0)
    assert allclose(tmp["x_2"][0], 2.0)
    assert allclose(tmp["x_1"][0], 1.0)
    tmp = stats.maximum()
    assert allclose(tmp["cstr"][0], -1.0)
    assert allclose(tmp["obj"][0], 21.0)
    assert allclose(tmp["x_2"][0], 11.0)
    assert allclose(tmp["x_1"][0], 10.0)
    tmp = stats.range()
    assert allclose(tmp["cstr"][0], 0.0)
    assert allclose(tmp["obj"][0], 18.0)
    assert allclose(tmp["x_2"][0], 9.0)
    assert allclose(tmp["x_1"][0], 9.0)


def test_empstats_mean(mc_datasets):
    (
        stats,
        _,
        _,
    ) = mc_datasets
    tmp = stats.mean()
    assert allclose(tmp["cstr"][0], -1.0)
    assert allclose(tmp["obj"][0], 12.0)
    assert allclose(tmp["x_2"][0], 6.5)
    assert allclose(tmp["x_1"][0], 5.5)


def test_empstats_std(mc_datasets):
    stats, stats_obj, _ = mc_datasets
    tmp = stats.standard_deviation()
    assert allclose(tmp["cstr"][0], 0.0)
    assert allclose(tmp["obj"][0], 5.744563, rtol=1e-06)
    assert allclose(tmp["x_2"][0], 2.872281, rtol=1e-06)
    assert allclose(tmp["x_1"][0], 2.872281, rtol=1e-06)
    tmp = stats.variance()
    assert allclose(tmp["cstr"][0], 0.0, rtol=1e-06)
    assert allclose(tmp["obj"][0], 33.0, rtol=1e-06)
    assert allclose(tmp["x_2"][0], 8.25, rtol=1e-06)
    assert allclose(tmp["x_1"][0], 8.25, rtol=1e-06)
    assert allclose(stats_obj.variance()["obj"][0], 33.0)


def test_empstats_prob(mc_datasets):
    stats, _, _ = mc_datasets
    value = {
        "cstr": array([3.0]),
        "obj": array([3.0]),
        "x_2": array([3.0]),
        "x_1": array([3.0]),
    }
    tmp = stats.probability(value)
    assert allclose(tmp["cstr"], 0.0)
    assert allclose(tmp["obj"], 1.0)
    assert allclose(tmp["x_2"], 0.9)
    assert allclose(tmp["x_1"], 0.8)
    tmp = stats.probability(value, False)
    assert allclose(tmp["cstr"], 1.0)
    assert allclose(tmp["obj"], 0.1)
    assert allclose(tmp["x_2"], 0.2)
    assert allclose(tmp["x_1"], 0.3)


def test_empstats_quant(mc_datasets):
    stats, stats_obj, _ = mc_datasets
    tmp = stats.quantile(0.5)
    assert allclose(tmp["cstr"][0], -1.0)
    assert allclose(tmp["obj"][0], 12.0)
    assert allclose(tmp["x_1"][0], 5.5)
    assert allclose(tmp["x_2"][0], 6.5)
    assert allclose(stats_obj.quantile(0.5)["obj"][0], 12.0)


def test_empstats_quart(mc_datasets):
    stats, stats_obj, _ = mc_datasets
    tmp = stats.quartile(2)
    assert allclose(tmp["cstr"][0], -1.0)
    assert allclose(tmp["obj"][0], 12.0)
    assert allclose(tmp["x_1"][0], 5.5)
    assert allclose(tmp["x_2"][0], 6.5)
    assert allclose(stats_obj.quartile(2)["obj"][0], 12.0)
    with pytest.raises(ValueError):
        stats.quartile(0.25)


def test_empstats_perc(mc_datasets):
    stats, stats_obj, _ = mc_datasets
    tmp = stats.percentile(50)
    assert allclose(tmp["cstr"][0], -1.0)
    assert allclose(tmp["obj"][0], 12.0)
    assert allclose(tmp["x_1"][0], 5.5)
    assert allclose(tmp["x_2"][0], 6.5)
    assert allclose(stats_obj.percentile(50)["obj"][0], 12.0)
    with pytest.raises(TypeError):
        stats.percentile(0.25)
    with pytest.raises(TypeError):
        stats.percentile(-1)


def test_empstats_med(mc_datasets):
    stats, stats_obj, _ = mc_datasets
    tmp = stats.median()
    assert allclose(tmp["cstr"][0], -1.0)
    assert allclose(tmp["obj"][0], 12.0)
    assert allclose(tmp["x_1"][0], 5.5)
    assert allclose(tmp["x_2"][0], 6.5)
    assert allclose(stats_obj.median()["obj"][0], 12.0)


def test_empstats_moment(mc_datasets):
    stats, _, _ = mc_datasets
    tmp = stats.moment(1)
    assert allclose(tmp["cstr"][0], 0.0)
    assert allclose(tmp["obj"][0], 0.0)
    assert allclose(tmp["x_1"][0], 0.0)
    assert allclose(tmp["x_2"][0], 0.0)
