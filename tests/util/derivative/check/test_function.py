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
"""Tests for FunctionJacobianChecker."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import load as np_load

from gemseo.core.function.array_function import ArrayFunction
from gemseo.util.derivative.check.function import FunctionJacobianChecker

# A linear function f(x) = A @ x with known constant Jacobian A.
_A = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
_X = array([1.0, 2.0, 3.0])
_X_1D = array([1.0])


# Module-level callables are required for multiprocessing (spawn on Windows).
def _linear_func(x):
    return _A @ x


def _linear_jac(x):
    return _A


@pytest.fixture(scope="module")
def checker() -> FunctionJacobianChecker:
    """Checker for a 2-output, 3-input linear function with exact Jacobian."""
    return FunctionJacobianChecker(
        ArrayFunction(_linear_func, name="linear", jac=_linear_jac)
    )


# ---------------------------------------------------------------------------
# parallel execution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_threading", [False, True])
def test_parallel(use_threading) -> None:
    """Check that parallel=True with processes or threads produces a correct result."""
    checker = FunctionJacobianChecker(
        ArrayFunction(_linear_func, name="linear", jac=_linear_jac)
    )
    assert checker.check(_X, n_processes=2, use_threading=use_threading)


# ---------------------------------------------------------------------------
# input_indices / output_indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_indices", [(), (0,), (1, 2), (0, 1, 2)])
def test_input_indices(checker, input_indices) -> None:
    """Check that restricting input components still passes for a correct Jacobian."""
    assert checker.check(_X, inputs=input_indices)


@pytest.mark.parametrize("output_indices", [(), (0,), (1,), (0, 1)])
def test_output_indices(checker, output_indices) -> None:
    """Check that restricting output components still passes for a correct Jacobian."""
    assert checker.check(_X, outputs=output_indices)


def test_input_and_output_indices(checker) -> None:
    """Check that combining both index selections works."""
    assert checker.check(_X, inputs=(0, 2), outputs=(1,))


def test_indices_catch_wrong_jacobian() -> None:
    """Check that a wrong Jacobian block is detected even when indices restrict it."""
    wrong_jac = _A.copy()
    wrong_jac[1, 2] += 10.0
    function = ArrayFunction(_linear_func, name="wrong", jac=lambda x: wrong_jac)
    checker = FunctionJacobianChecker(function)

    # Full check catches the error (returns False by default).
    assert not checker.check(_X)

    # Check restricted to the wrong block also catches it.
    assert not checker.check(_X, inputs=(2,), outputs=(1,))

    # Check that does NOT include the wrong block passes.
    assert checker.check(_X, inputs=(0, 1), outputs=(0,))


# ---------------------------------------------------------------------------
# reference_jacobian_path / save_reference_jacobian
# ---------------------------------------------------------------------------


def test_save_reference_jacobian(checker, tmp_wd) -> None:
    """Check that save_reference_jacobian writes the approximated Jacobian to disk."""
    path = "reference.npy"
    assert checker.check(_X, reference_jacobian_path=path, save_reference_jacobian=True)
    saved = np_load(path)
    assert saved.shape == _A.shape


def test_load_reference_jacobian(checker, tmp_wd) -> None:
    """Check that a saved reference Jacobian can be reloaded for comparison."""
    path = "reference.npy"
    assert checker.check(_X, reference_jacobian_path=path, save_reference_jacobian=True)
    assert checker.check(_X, reference_jacobian_path=path)


def test_load_reference_jacobian_with_indices(checker, tmp_wd) -> None:
    """Check that indices are applied correctly when loading a saved reference."""
    path = "reference.npy"
    assert checker.check(_X, reference_jacobian_path=path, save_reference_jacobian=True)
    assert checker.check(_X, reference_jacobian_path=path, inputs=(0, 1), outputs=(0,))
