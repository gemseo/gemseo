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
from gemseo.algos.opt.opt_factory import OptimizersFactory


def test_is_available_error():
    assert not OptimizersFactory().is_available("None")


def test_init_library_error():
    OptimizersFactory().create("L-BFGS-B")
    with pytest.raises(
        ImportError, match="No algorithm or library of algorithms named "
    ):
        OptimizersFactory().create("idontexist")


def test_is_scipy_available():
    assert OptimizersFactory().is_available("ScipyOpt")
    assert "SLSQP" in OptimizersFactory().algorithms
