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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.formulations.disciplinary_opt import DisciplinaryOpt
from gemseo.problems.sobieski.wrappers import (
    SobieskiMission,
    SobieskiProblem,
    SobieskiStructure,
)

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class Test_DisciplinaryOpt(unittest.TestCase):
    """ """

    def test_multiple_disc(self):
        """ """
        ds = SobieskiProblem().read_design_space()
        dopt = DisciplinaryOpt([SobieskiStructure(), SobieskiMission()], "y_4", ds)
        dopt.get_expected_dataflow()
        dopt.get_expected_workflow()

    def test_init(self):
        """ """
        sm = SobieskiMission()
        ds = SobieskiProblem().read_design_space()
        dopt = DisciplinaryOpt([sm], "y_4", ds)
        assert dopt.get_expected_dataflow() == []
        assert dopt.get_expected_workflow().sequence_list[0].discipline == sm
        assert len(dopt.get_expected_workflow().sequence_list) == 1
