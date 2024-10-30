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

import unittest
from pathlib import Path

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post.factory import PostFactory
from gemseo.post.opt_history_view import OptHistoryView

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"


class TestBasePostFactory(unittest.TestCase):
    """"""

    def test_is_available(self) -> None:
        """"""
        factory = PostFactory()
        assert factory.is_available("OptHistoryView")
        assert not factory.is_available("TOTO")
        self.assertRaises(ImportError, factory.create, None, "toto")

    def test_post(self) -> None:
        assert PostFactory().is_available("GradientSensitivity")
        assert PostFactory().is_available("Correlations")

    def test_execute_from_hdf(self) -> None:
        opt_problem = OptimizationProblem.from_hdf(POWER2)
        post = PostFactory().execute(
            opt_problem, post_name="OptHistoryView", save=False
        )
        assert isinstance(post, OptHistoryView)
