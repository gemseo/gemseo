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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.mlearning.regression.quality.me_measure import MEMeasure


@pytest.fixture(scope="module")
def me() -> MEMeasure:
    """An MEMeasure with mocked argument values passed at instantiation."""
    return MEMeasure("mocked_algo", fit_transformers="mocked_fit_transformers")


OUTPUTS = array([[0, 1, 0], [1, 0, 2]])
PREDICTIONS = array([[0, 1, 0], [1, 1, 0]])


def test_init(me):
    """Check that the arguments are correctly used at instantiation."""
    assert me.algo == "mocked_algo"
    assert me._fit_transformers == "mocked_fit_transformers"


def test_compute_measure(me):
    """Check _compute with default value for multioutput."""
    assert_equal(me._compute_measure(OUTPUTS, PREDICTIONS), array([0.0, 1.0, 2.0]))


@pytest.mark.parametrize(
    ("multioutput", "expected"), [(False, 2.0), (True, array([0.0, 1.0, 2.0]))]
)
def test_compute_measure_with_multioutput(me, multioutput, expected):
    """Check _compute with multioutput."""
    assert_equal(
        me._compute_measure(OUTPUTS, PREDICTIONS, multioutput=multioutput), expected
    )
