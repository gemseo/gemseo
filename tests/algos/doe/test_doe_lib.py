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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

import pytest

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.problems.analytical.power_2 import Power2

FACTORY = DOEFactory()


@pytest.fixture
def doe():
    pytest.mark.skipif(
        FACTORY.is_available("PyDOE"), reason="skipped because PyDOE is missing"
    )
    return FACTORY.create("PyDOE")


def test_fail_sample(doe):
    problem = Power2(exception_error=True)
    doe.execute(problem, "lhs", n_samples=4)


def test_evaluate_samples(doe):
    problem = Power2()
    doe.execute(problem, "fullfact", n_samples=2, wait_time_between_samples=1)


@pytest.mark.skip_under_windows
def test_evaluate_samples_multiproc(doe):
    problem = Power2()
    doe.execute(
        problem,
        "fullfact",
        n_samples=2,
        n_processes=2,
        wait_time_between_samples=1,
    )
