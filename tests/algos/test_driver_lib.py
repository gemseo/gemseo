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

import pytest

from gemseo.algos.driver_lib import DriverLib


class MyDriver(DriverLib):
    pass


def test_max_iter_fail():
    """Check that a ValueError is raised for an invalid `max_iter` input."""

    MyDriver()._pre_run(None, None)
    with pytest.raises(ValueError, match="max_iter must be >=1, got -1"):
        MyDriver().init_iter_observer(max_iter=-1, message="message")


def test_no_algo_fail():
    """Check that a ValueError is raised when no algorithm name is set."""

    with pytest.raises(
        ValueError,
        match="Algorithm name must be either passed as "
        "argument or set by the attribute self.algo_name",
    ):
        MyDriver().execute(None)


def test_grammar_fail():
    """Check that a ValueError is raised when the grammar file is not found."""

    with pytest.raises(
        ValueError,
        match=(
            "Neither the options grammar file .+ for the algorithm 'unknown' "
            "nor the options grammar file .+ for the library 'DriverLib' "
            "has been found."
        ),
    ):
        DriverLib().init_options_grammar("unknown")


def test_require_grad():
    """Check that an error is raised when a particular gradient method is not given."""

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

    with pytest.raises(ValueError, match="Algorithm toto is not available."):
        MyDriver().is_algo_requires_grad("toto")

    assert MyDriver().is_algo_requires_grad("SLSQP")
