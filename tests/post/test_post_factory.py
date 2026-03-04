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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path

import pytest

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post import OptHistoryView_Settings
from gemseo.post.factory import POST_FACTORY
from gemseo.post.opt_history_view import OptHistoryView

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"


def test_is_available() -> None:
    """"""
    assert POST_FACTORY.is_available("OptHistoryView")
    assert not POST_FACTORY.is_available("TOTO")
    with pytest.raises(ImportError):
        POST_FACTORY.create(None, "toto")


def test_post() -> None:
    assert POST_FACTORY.is_available("GradientSensitivity")
    assert POST_FACTORY.is_available("Correlations")


def test_execute_from_hdf() -> None:
    opt_problem = OptimizationProblem.from_hdf(POWER2)
    post = POST_FACTORY.execute(opt_problem, OptHistoryView_Settings(save=False))
    assert isinstance(post, OptHistoryView)
