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
# INITIAL AUTHORS - initial API and implementation and/or
#                   initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import os
from os.path import dirname
from os.path import join

import pytest
from gemseo.caches.hdf5_cache import HDF5Cache
from gemseo.problems.scalable.data_driven.problem import ScalableProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure

DIRNAME = dirname(__file__)
HDF_CACHE_PATH = join(DIRNAME, "dataset.hdf5")

N_SAMPLES = 10


@pytest.fixture(scope="module")
def scalable_problem():
    design_variables = ["x_shared", "x_1", "x_2", "x_3"]
    objective_function = "y_4"
    ineq_constraints = ["g_1", "g_2"]
    eq_constraints = ["g_3"]
    aero = SobieskiAerodynamics()
    propu = SobieskiPropulsion()
    struct = SobieskiStructure()
    mission = SobieskiMission()
    disciplines = [aero, propu, struct, mission]
    disc_names = [disc.name for disc in disciplines]
    datasets = [
        HDF5Cache(HDF_CACHE_PATH, disc).export_to_dataset() for disc in disc_names
    ]
    scalpbm = ScalableProblem(
        datasets, design_variables, objective_function, eq_constraints, ineq_constraints
    )
    return scalpbm


def test_print(scalable_problem):
    assert "Sizes" in str(scalable_problem)


def test_plot_n2_chart(scalable_problem, tmp_wd):
    """"""
    scalable_problem.plot_n2_chart()
    assert os.path.exists("n2.pdf")


def test_plot_coupling_graph(scalable_problem, tmp_wd):
    """"""
    scalable_problem.plot_coupling_graph()
    assert os.path.exists("coupling_graph.pdf")


def test_plot_1d_interpolations(scalable_problem, tmp_wd):
    """"""
    files = scalable_problem.plot_1d_interpolations(directory=str(tmp_wd))
    assert len(files) > 0
    for fname in files:
        assert os.path.exists(fname)


def test_plot_dependencies(scalable_problem, tmp_wd):
    """"""
    files = scalable_problem.plot_dependencies(directory=str(tmp_wd))
    assert len(files) > 0
    for fname in files:
        assert os.path.exists(fname)


def test_create_scenario(scalable_problem):
    """"""
    scalable_problem.create_scenario()


def test_statistics(scalable_problem):
    """"""
    scalable_problem.create_scenario()
    scalable_problem.exec_time()
    scalable_problem.n_calls
    scalable_problem.n_calls_linearize
    scalable_problem.scenario.execute({"algo": "SLSQP", "max_iter": 100})
    scalable_problem.status
