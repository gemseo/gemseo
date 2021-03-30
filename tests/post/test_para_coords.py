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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, unicode_literals

import unittest
from os import remove
from os.path import dirname, exists, join

from future import standard_library
from numpy import ones

from gemseo import SOFTWARE_NAME
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.api import configure_logger
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.third_party.junitxmlreq import link_to

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)

POWER2 = join(dirname(__file__), "power2_opt_pb.h5")


class Test_ParaCoords(unittest.TestCase):
    """ """

    @link_to("Req-VIZ-1", "Req-VIZ-1.1", "Req-VIZ-1.2", "Req-VIZ-2", "Req-VIZ-1.7")
    def test_scatter(self):
        """ """
        if PostFactory().is_available("ParallelCoordinates"):
            n = 10
            problem = Rosenbrock(n)
            problem.x_0 = ones(n) * 0.99
            OptimizersFactory().execute(problem, "SLSQP")
            post = PostFactory().execute(
                problem,
                "ParallelCoordinates",
                save=True,
                figsize_x=12,
                file_path="para_coords1",
            )
            assert len(post.output_files) == 2
            for outf in post.output_files:
                assert exists(outf)
                remove(outf)

    @link_to(
        "Req-VIZ-1",
        "Req-VIZ-1.1",
        "Req-VIZ-1.2",
        "Req-VIZ-2",
        "Req-VIZ-5",
        "Req-VIZ-1.7",
    )
    def test_scatter_load(self):
        """ """
        if PostFactory().is_available("ParallelCoordinates"):
            post = PostFactory().execute(
                POWER2, "ParallelCoordinates", save=True, figsize_x=6
            )
            assert len(post.output_files) == 2
            for outf in post.output_files:
                assert exists(outf)
                remove(outf)
