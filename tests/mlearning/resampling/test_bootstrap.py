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
"""Test for the module bootstrap."""

from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo import SEED
from gemseo.mlearning.resampling.bootstrap import Bootstrap


@pytest.fixture(scope="module")
def bootstrap(sample_indices) -> Bootstrap:
    """A bootsrap."""
    return Bootstrap(sample_indices)


def test_default_properties(bootstrap, sample_indices):
    """Check the default values of the properties."""
    assert_equal(bootstrap.sample_indices, sample_indices)
    assert bootstrap.seed == SEED
    assert bootstrap.n_replicates == 100
    assert len(bootstrap.splits) == 100


def test_properties_with_n_replicates(sample_indices):
    """Check that the number of folds depends on n_replicates."""
    bootstrap = Bootstrap(sample_indices, n_replicates=2)
    assert bootstrap.n_replicates == 2
    assert len(bootstrap.splits) == 2


@pytest.mark.parametrize("stack_predictions", [False, True])
def test_stacked_predictions(bootstrap, stack_predictions):
    """Check the stacked_predictions argument of the method execute."""
    predictions = [array([[1, 2], [3, 4]]), array([[1, 2], [3, 4]])]
    result = bootstrap._post_process_predictions(predictions, (3, 2), stack_predictions)
    if stack_predictions:
        assert result.shape == (4, 2)
    else:
        assert result == result
