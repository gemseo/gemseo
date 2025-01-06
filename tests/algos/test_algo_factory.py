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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author: Remi Lafage
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest

from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.algos.opt.factory import OptimizationLibraryFactory


def test_is_available_error() -> None:
    assert not OptimizationLibraryFactory().is_available("None")


def test_create_ok() -> None:
    """Verify that an existing algorithm can be created."""
    algo = OptimizationLibraryFactory().create("L-BFGS-B")
    assert algo._algo_name == "L-BFGS-B"
    assert "max_iter" in algo._validate_settings()


def test_create_ko() -> None:
    """Verify that an error is raised when trying to create an unknown algorithm."""
    with pytest.raises(
        ValueError,
        match=(
            r"No algorithm named idontexist is available; available algorithms are .*"
        ),
    ):
        OptimizationLibraryFactory().create("idontexist")


def test_is_scipy_available() -> None:
    assert OptimizationLibraryFactory().is_available("ScipyOpt")
    assert "SLSQP" in OptimizationLibraryFactory().algorithms


def test_solver_factory_cache() -> None:
    """Verify the caching of the solver factory."""
    factory = LinearSolverLibraryFactory(use_cache=True)
    lib1 = factory.create("DEFAULT")
    lib2 = factory.create("DEFAULT")
    assert lib2 is lib1

    # A new instance has a different cache.
    factory = LinearSolverLibraryFactory(use_cache=True)
    lib1_bis = factory.create("DEFAULT")
    lib2_bis = factory.create("DEFAULT")
    assert lib2_bis is lib1_bis
    assert lib2_bis is not lib2
    assert lib1_bis is not lib1


def test_clear_lib_cache() -> None:
    """Verify clearing the lib cache."""
    factory = LinearSolverLibraryFactory(use_cache=True)
    lib1 = factory.create("DEFAULT")
    factory.clear_lib_cache()
    lib2 = factory.create("DEFAULT")
    assert lib1 is not lib2
