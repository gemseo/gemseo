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

import pytest
from numpy import array
from numpy import int32

from gemseo.mda.base_mda_solver import BaseMDASolver
from gemseo.mda.base_mda_solver_settings import BaseMDASolverSettings
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.utils.testing.helpers import concretize_classes


class _BaseMDASolver(BaseMDASolver):
    """Concretizable BaseMDASolver class for test purpose."""

    Settings = BaseMDASolverSettings
    """The Pydantic model for the settings."""


@pytest.fixture
def base_mda_solver() -> _BaseMDASolver:
    with concretize_classes(_BaseMDASolver):
        return _BaseMDASolver([Sellar1(), Sellar2()])


def test_integer_casting(base_mda_solver):
    """Tests residual computation with integers."""
    base_mda_solver._set_resolved_variables(["y_1", "y_2"])

    base_mda_solver.io.data["y_1"] = array([2], dtype=int32)
    base_mda_solver.io.data["y_2"] = array([2.0])

    base_mda_solver._compute_residuals({"y_1": array([1.0]), "y_2": array([1.0])})

    assert base_mda_solver._current_residuals["y_1"] == array([1.0])
    assert base_mda_solver._current_residuals["y_2"] == array([1.0])
