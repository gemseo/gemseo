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
#       :author: Pierre-Jean Barjhoux
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, unicode_literals

import unittest
from os import remove
from os.path import dirname, exists, join

import numpy as np
from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import configure_logger
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.third_party.junitxmlreq import link_to

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)
POWER2 = join(dirname(__file__), "power2_opt_pb.h5")


class Test_ObjConstrHist(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(cls):
        problem = Power2()
        problem.x_0 = np.ones(3) * 50
        problem.l_bounds = -np.ones(3)
        problem.u_bounds = np.ones(3) * 50
        OptimizersFactory().execute(problem, "SLSQP")
        cls.problem = problem
        cls.factory = PostFactory()

    @link_to("Req-VIZ-1", "Req-VIZ-1.6", "Req-VIZ-4")
    def test_objconstr(self):
        """ """
        if self.factory.is_available("ObjConstrHist"):
            post = self.factory.execute(
                self.problem,
                "ObjConstrHist",
                save=True,
                show=False,
                file_path="ObjConstrHist1",
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)
                remove(outf)

    @link_to("Req-VIZ-1", "Req-VIZ-1.6", "Req-VIZ-4")
    def test_objconstr_load(self):
        """ """
        if self.factory.is_available("ObjConstrHist"):
            problem = OptimizationProblem.import_hdf(file_path=POWER2)
            post = self.factory.execute(
                problem,
                "ObjConstrHist",
                save=True,
                show=False,
                file_path="ObjConstrHist2",
                constr_names=["ineq1", "ineq2"],
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)
                remove(outf)

    @link_to("Req-VIZ-1", "Req-VIZ-1.6", "Req-VIZ-4")
    def test_objconstr_name(self):
        """ """
        if self.factory.is_available("ObjConstrHist"):
            post = self.factory.execute(
                self.problem,
                "ObjConstrHist",
                file_path="ObjConstrHist3",
                save=True,
                show=False,
                constr_names=["eq"],
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)
                remove(outf)

    @link_to("Req-VIZ-1", "Req-VIZ-1.6", "Req-VIZ-4")
    def test_objconstr_name_load(self):
        """ """
        if self.factory.is_available("ObjConstrHist"):
            problem = OptimizationProblem.import_hdf(file_path=POWER2)
            post = self.factory.execute(
                problem,
                "ObjConstrHist",
                save=True,
                show=False,
                constr_names=["ineq1", "ineq2"],
                file_path="ObjConstrHist4",
            )
            assert len(post.output_files) == 1
            for outf in post.output_files:
                assert exists(outf)
                remove(outf)
