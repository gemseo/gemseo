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

from gemseo.mlearning.regression.quality.mae_measure import MAEMeasure


@pytest.fixture(scope="module")
def mae() -> MAEMeasure:
    """An MAEMeasure with mocked argument values passed at instantiation."""
    return MAEMeasure("mocked_algo", fit_transformers="mocked_fit_transformers")


OUTPUTS = array([[0, 1], [1, 0]])
PREDICTIONS = array([[0, 1], [1, 1]])


def test_init(mae):
    """Check that the arguments are correctly used at instantiation."""
    assert mae.algo == "mocked_algo"
    assert mae._fit_transformers == "mocked_fit_transformers"


def test_compute_measure(mae):
    """Check _compute with default value for multioutput."""
    assert_equal(mae._compute_measure(OUTPUTS, PREDICTIONS), array([0.0, 0.5]))


@pytest.mark.parametrize(
    ("multioutput", "expected"), [(False, 0.25), (True, array([0.0, 0.5]))]
)
def test_compute_measure_with_multioutput(mae, multioutput, expected):
    """Check _compute with multioutput."""
    assert_equal(
        mae._compute_measure(OUTPUTS, PREDICTIONS, multioutput=multioutput), expected
    )
