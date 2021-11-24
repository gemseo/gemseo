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
from unittest import mock

import pytest
from numpy import arange, array

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.doe_lib import DOELibrary
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


def test_phip_criteria():
    """Check that the phi-p criterion is well implemented."""
    power = 3.0
    samples = array([[0.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
    expected = sum([val ** (-power) for val in [2.0, 3.0, 1.0]]) ** (1.0 / power)
    assert DOELibrary.compute_phip_criteria(samples, power) == expected


@pytest.fixture(scope="module")
def variables_space():
    """A mock design space."""
    design_space = mock.Mock()
    design_space.dimension = 2
    design_space.untransform_vect = mock.Mock(return_value=arange(6).reshape((3, 2)))
    return design_space


def test_compute_doe_transformed(doe, variables_space):
    """Check the computation of a transformed DOE in a variables space."""
    doe.algo_name = "lhs"
    points = doe.compute_doe(variables_space, size=3, unit_sampling=True)
    assert points.shape == (3, 2)
    assert points.max() <= 1.0
    assert points.min() >= 0.0
    variables_space.untransform_vect.assert_not_called()


def test_compute_doe_nontransformed(doe, variables_space):
    """Check the computation of a non-transformed DOE in a variables space."""
    doe.algo_name = "lhs"
    points = doe.compute_doe(variables_space, size=3)
    assert points.shape == (3, 2)
    variables_space.untransform_vect.assert_called_once()
