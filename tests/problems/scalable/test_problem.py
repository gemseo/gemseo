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
# INITIAL AUTHORS - initial API and implementation and/or
#                   initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, unicode_literals

import os
from os.path import dirname, join

import pytest
from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.caches.hdf5_cache import HDF5Cache
from gemseo.problems.scalable.problem import ScalableProblem
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)

standard_library.install_aliases()

configure_logger(SOFTWARE_NAME)

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
    caches = [HDF5Cache(HDF_CACHE_PATH, disc) for disc in disc_names]
    scalpbm = ScalableProblem(
        caches, design_variables, objective_function, eq_constraints, ineq_constraints
    )
    return scalpbm


@pytest.fixture(scope="module")
def working_directory(tmpdir_factory):
    return str(tmpdir_factory.mktemp("working_directory"))


def test_print(scalable_problem):
    assert "Sizes" in str(scalable_problem)


def test_plot_n2_chart(scalable_problem):
    """ """
    scalable_problem.plot_n2_chart(show=False, save=True)
    assert os.path.exists("n2.pdf")
    os.remove("n2.pdf")


def test_plot_coupling_graph(scalable_problem):
    """ """
    scalable_problem.plot_coupling_graph()
    assert os.path.exists("coupling_graph.pdf")
    os.remove("coupling_graph.pdf")


def test_plot_1d_interpolations(scalable_problem, working_directory):
    """ """
    files = scalable_problem.plot_1d_interpolations(
        show=False, save=True, directory=working_directory
    )
    assert len(files) > 0
    for fname in files:
        assert os.path.exists(fname)


def test_plot_dependencies(scalable_problem, working_directory):
    """ """
    files = scalable_problem.plot_dependencies(
        show=False, save=True, directory=working_directory
    )
    assert len(files) > 0
    for fname in files:
        assert os.path.exists(fname)


def test_create_scenario(scalable_problem):
    """ """
    scalable_problem.create_scenario()


def test_statistics(scalable_problem):
    """ """
    scalable_problem.create_scenario()
    scalable_problem.exec_time()
    scalable_problem.n_calls
    scalable_problem.n_calls_linearize
    scalable_problem.scenario.execute({"algo": "SLSQP", "max_iter": 100})
    scalable_problem.status
