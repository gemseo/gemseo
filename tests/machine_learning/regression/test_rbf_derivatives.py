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
"""Test the derivatives of the radial basis function regression module."""

from __future__ import annotations

import pytest

from gemseo.machine_learning.regression.model._rbf_derivatives import KERNEL_DERIVATIVES
from gemseo.machine_learning.regression.model._rbf_derivatives import (
    BaseKernelDerivative,
)
from gemseo.machine_learning.regression.model.rbf_settings import RBF


@pytest.mark.parametrize("kernel", RBF)
def test_kernel_derivatives(kernel) -> None:
    """Check that every RBF kernel has a derivative."""
    assert issubclass(KERNEL_DERIVATIVES[kernel], BaseKernelDerivative)
