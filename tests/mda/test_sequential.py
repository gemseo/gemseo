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
#
# Copyright 2024 Capgemini
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from functools import partial

import pytest
from numpy import allclose
from numpy import array
from numpy import inf

from gemseo.mda.base_mda import BaseMDA
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.newton_raphson import MDANewtonRaphson
from gemseo.mda.sequential_mda import MDASequential
from gemseo.problems.mdo.scalable.linear.linear_discipline import LinearDiscipline
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.utils import get_y_opt

from .utils import generate_parallel_doe

allclose_ = partial(allclose, rtol=1e-6, atol=0.0)


@pytest.fixture
def mda_sequential() -> MDASequential:
    """A sequential MDA with Jacobi and Newton-Raphson on the Sellar 1/2 disciplines."""
    disciplines = [Sellar1(), Sellar2()]
    return MDASequential(
        disciplines,
        [
            MDAJacobi(disciplines, max_mda_iter=2),
            MDANewtonRaphson(disciplines),
        ],
    )


def test_execution(mda_sequential) -> None:
    """Test the execution."""
    mda_sequential.execute()

    assert allclose_(array([0.8, 1.8]), get_y_opt(mda_sequential))
    assert mda_sequential.normed_residual <= 1e-6

    mda_1, mda_2 = mda_sequential.mda_sequence
    assert mda_sequential.normed_residual == mda_2.normed_residual
    assert mda_sequential._current_iter == mda_1._current_iter + mda_2._current_iter


def test_parallel_doe() -> None:
    """Test the execution of GaussSeidel in parallel."""
    obj = generate_parallel_doe(main_mda_settings={"inner_mda_name": "MDAGSNewton"})
    assert allclose_(array([-obj]), array([608.175]), atol=1e-3)


def test_scaling_setter(mda_sequential) -> None:
    """Test the setting of the `scaling` attribute of a sequential MDA.

    Check that the change is propagated to the sub-MDAs.
    """
    mda_sequential.scaling = BaseMDA.ResidualScaling.NO_SCALING
    assert mda_sequential.scaling == BaseMDA.ResidualScaling.NO_SCALING
    for sub_mda in mda_sequential.mda_sequence:
        assert sub_mda.scaling == BaseMDA.ResidualScaling.NO_SCALING


def test_cascading_of_log_convergence(mda_sequential) -> None:
    """Test the cascading of the log_convergence setting."""
    mda_sequential.settings.log_convergence = False
    assert not mda_sequential.settings.log_convergence
    for sub_mda in mda_sequential.mda_sequence:
        assert not sub_mda.settings.log_convergence

    mda_sequential.settings.log_convergence = True
    assert mda_sequential.settings.log_convergence
    for sub_mda in mda_sequential.mda_sequence:
        assert sub_mda.settings.log_convergence


def test_set_bounds():
    """Test that bounds are properly dispatched to inner-MDAs."""
    disciplines = [
        LinearDiscipline("A", ["x", "b"], ["a"]),
        LinearDiscipline("B", ["a"], ["b", "y"]),
    ]
    mda = MDASequential(
        disciplines=disciplines,
        mda_sequence=[
            MDAJacobi(disciplines, max_mda_iter=2),
            MDANewtonRaphson(disciplines),
        ],
    )

    lower_bound = -array([1.0])
    upper_bound = array([1.0])

    mda.set_bounds({
        "a": (lower_bound, None),
        "b": (2.0 * lower_bound, upper_bound),
    })

    mda.execute()

    for sub_mda in mda.mda_sequence:
        assert (sub_mda.lower_bound_vector == array([-1.0, -2.0])).all()
        assert (sub_mda.upper_bound_vector == array([+inf, 1.0])).all()

        assert (sub_mda._sequence_transformer.lower_bound == array([-1.0, -2.0])).all()
        assert (sub_mda._sequence_transformer.upper_bound == array([+inf, 1.0])).all()
