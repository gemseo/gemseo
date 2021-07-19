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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Driver library tests."""

from __future__ import division, unicode_literals

import unittest

from gemseo.algos.driver_lib import DriverLib


class TestDriverLib(unittest.TestCase):
    def test_fail_messages(self):
        class MyDriver(DriverLib):
            pass

        MyDriver()._pre_run(None, None)
        self.assertRaises(
            ValueError, MyDriver().init_iter_observer, max_iter=-1, message="message"
        )

        self.assertRaises(ValueError, MyDriver().execute, None)

        self.assertRaises(ValueError, DriverLib().init_options_grammar, "unknown")

    def test_require_grad(self):
        class MyDriver(DriverLib):
            def __init__(self):
                super(MyDriver, self).__init__()
                self.lib_dict = {
                    "SLSQP": {
                        DriverLib.INTERNAL_NAME: "SLSQP",
                        DriverLib.REQUIRE_GRAD: True,
                        DriverLib.POSITIVE_CONSTRAINTS: True,
                        DriverLib.HANDLE_EQ_CONS: True,
                        DriverLib.HANDLE_INEQ_CONS: True,
                    }
                }

        self.assertRaises(ValueError, MyDriver().is_algo_requires_grad, "toto")
        assert MyDriver().is_algo_requires_grad("SLSQP")
